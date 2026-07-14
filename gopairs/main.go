// gopairs -- mine BM25 hard negatives for stage-2 contrastive training.
//
// The Python version (rank_bm25) calls get_scores() per query, which scores EVERY document: O(N^2)
// dense. Measured at 1,500 passages it takes 50s, so 20,000 passages is ~178x that -- about 2.5
// hours per source, ~10 hours for all four. Unusable.
//
// Two fixes, and the algorithmic one matters more than the language:
//
//  1. INVERTED INDEX. Only documents that actually share a term with the query are ever touched.
//     Legal text has a long-tailed vocabulary, so most doc/query pairs share nothing and dense
//     scoring spends nearly all its time adding zeros.
//  2. GOROUTINES over queries, one score accumulator each.
//
// Loading the HF datasets stays in Python (it needs `datasets`); only the mining -- pure computation
// with no Python library in it -- moves here.
//
// NOTE ON PARITY: this does not reproduce rank_bm25 bit-for-bit. rank_bm25's BM25Okapi uses a
// negative-IDF correction (epsilon * average_idf); we use the standard +1-inside-log IDF, which
// cannot go negative. Both are valid BM25 variants, and for hard-negative mining only the RANKING of
// candidates matters, not the absolute scores -- so the difference is immaterial. Anything that
// depended on exact score values would need the Python version.
//
// Input  (jsonl): {"id":int, "text":str, "label":str, "source":str}
// Output (jsonl): {"id":int, "negatives":[int,...]}
//
//	go build -o gopairs . && ./gopairs -in ../data/pairs/_mine_in.jsonl -out ../data/pairs/_mine_out.jsonl -k 2
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
)

const (
	k1 = 1.5
	b  = 0.75
)

// The index is built over POSITIVES (the passages we retrieve), but queried with the ANCHOR (the
// query text). They are different strings -- conflating them would mine negatives for the wrong side.
type doc struct {
	ID       int    `json:"id"`
	Anchor   string `json:"anchor"`
	Positive string `json:"positive"`
	Label    string `json:"label"`
	Source   string `json:"source"`
}

type out struct {
	ID        int   `json:"id"`
	Negatives []int `json:"negatives"`
}

func tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

type posting struct {
	doc int
	tf  float64
}

// index is a per-source BM25 inverted index.
type index struct {
	postings map[string][]posting
	idf      map[string]float64
	lenNorm  []float64 // k1*(1-b+b*|D|/avgdl), precomputed per doc
	docs     []doc
}

func build(docs []doc) *index {
	n := len(docs)
	ix := &index{postings: map[string][]posting{}, idf: map[string]float64{},
		lenNorm: make([]float64, n), docs: docs}

	tfs := make([]map[string]float64, n)
	total := 0.0
	for i, d := range docs {
		toks := tokenize(d.Positive) // index the PASSAGES
		m := make(map[string]float64, len(toks))
		for _, t := range toks {
			m[t]++
		}
		tfs[i] = m
		total += float64(len(toks))
	}
	avgdl := total / float64(n)

	df := map[string]int{}
	for i, m := range tfs {
		dl := 0.0
		for _, c := range m {
			dl += c
		}
		ix.lenNorm[i] = k1 * (1 - b + b*dl/avgdl)
		for t, c := range m {
			ix.postings[t] = append(ix.postings[t], posting{i, c})
			df[t]++
		}
	}
	for t, f := range df {
		// +1 inside the log: IDF can never go negative, so no epsilon correction is needed.
		ix.idf[t] = math.Log(1 + (float64(n)-float64(f)+0.5)/(float64(f)+0.5))
	}
	return ix
}

// topNegatives returns the k highest-BM25 docs that are NOT valid positives for the query.
//
// banned is the TEXT-level guard: the same clause text can appear under more than one label, so
// "different label" is not sufficient -- a different-label candidate may still be text that is a
// legitimate positive for this anchor's label. Training the model to push it away would be actively
// wrong, which is the exact failure this guard exists to prevent.
func (ix *index) topNegatives(qi, k int, banned map[string]struct{}, rng *rand.Rand,
	scores []float64, touched []int) []int {
	for _, d := range touched {
		scores[d] = 0
	}
	touched = touched[:0]

	for t := range termFreq(ix.docs[qi].Anchor) { // query with the ANCHOR
		post, ok := ix.postings[t]
		if !ok {
			continue
		}
		idf := ix.idf[t]
		for _, p := range post {
			if scores[p.doc] == 0 {
				touched = append(touched, p.doc)
			}
			scores[p.doc] += idf * (p.tf * (k1 + 1)) / (p.tf + ix.lenNorm[p.doc])
		}
	}

	type cand struct {
		id int
		s  float64
	}
	cands := make([]cand, 0, len(touched))
	qLabel := ix.docs[qi].Label
	for _, d := range touched {
		if d == qi || ix.docs[d].Label == qLabel {
			continue
		}
		if _, bad := banned[ix.docs[d].Positive]; bad {
			continue
		}
		cands = append(cands, cand{d, scores[d]})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].s > cands[j].s })

	res := make([]int, 0, k)
	seen := map[int]struct{}{}
	for i := 0; i < len(cands) && len(res) < k; i++ {
		res = append(res, ix.docs[cands[i].id].ID)
		seen[cands[i].id] = struct{}{}
	}

	// Pad with random draws if BM25 could not find enough clean candidates (short anchors can share
	// no terms with anything). Same fallback as the Python reference.
	n := len(ix.docs)
	for tries := 0; len(res) < k && tries < 50; tries++ {
		j := rng.Intn(n)
		if j == qi || ix.docs[j].Label == qLabel {
			continue
		}
		if _, bad := banned[ix.docs[j].Positive]; bad {
			continue
		}
		if _, dup := seen[j]; dup {
			continue
		}
		seen[j] = struct{}{}
		res = append(res, ix.docs[j].ID)
	}
	return res
}

func termFreq(s string) map[string]float64 {
	m := map[string]float64{}
	for _, t := range tokenize(s) {
		m[t]++
	}
	return m
}

func main() {
	in := flag.String("in", "../data/pairs/_mine_in.jsonl", "")
	outPath := flag.String("out", "../data/pairs/_mine_out.jsonl", "")
	k := flag.Int("k", 2, "hard negatives per anchor")
	flag.Parse()

	f, err := os.Open(*in)
	if err != nil {
		panic(err)
	}
	var docs []doc
	sc := bufio.NewReaderSize(f, 8<<20)
	for {
		// ReadBytes, never bufio.Scanner: Scanner silently returns false on an over-long line and
		// reports SUCCESS. That bug truncated legal_regulations to 36 of 88,804 docs during dedup.
		line, err := sc.ReadBytes('\n')
		if len(line) > 1 {
			var d doc
			if json.Unmarshal(line, &d) == nil {
				docs = append(docs, d)
			}
		}
		if err != nil {
			break
		}
	}
	f.Close()
	fmt.Printf("loaded %d passages\n", len(docs))

	// mine WITHIN each source: a contract clause is not a meaningful negative for a financial-QA
	// question -- it would be trivially separable, i.e. useless as a hard negative.
	bySource := map[string][]doc{}
	for _, d := range docs {
		bySource[d.Source] = append(bySource[d.Source], d)
	}

	of, err := os.Create(*outPath)
	if err != nil {
		panic(err)
	}
	w := bufio.NewWriterSize(of, 8<<20)
	enc := json.NewEncoder(w)
	var wmu sync.Mutex

	t0 := time.Now()
	for src, sdocs := range bySource {
		ts := time.Now()
		ix := build(sdocs)

		// text-level guard set, per label, within this source
		textsOfLabel := map[string]map[string]struct{}{}
		for _, d := range sdocs {
			if textsOfLabel[d.Label] == nil {
				textsOfLabel[d.Label] = map[string]struct{}{}
			}
			textsOfLabel[d.Label][d.Positive] = struct{}{}
		}

		n := len(sdocs)
		workers := runtime.NumCPU()
		var wg sync.WaitGroup
		chunk := (n + workers - 1) / workers
		for wkr := 0; wkr < workers; wkr++ {
			lo, hi := wkr*chunk, min((wkr+1)*chunk, n)
			if lo >= hi {
				continue
			}
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				scores := make([]float64, n)
				touched := make([]int, 0, 4096)
				rng := rand.New(rand.NewSource(int64(lo)))
				for i := lo; i < hi; i++ {
					neg := ix.topNegatives(i, *k, textsOfLabel[sdocs[i].Label], rng, scores, touched)
					wmu.Lock()
					enc.Encode(out{ID: sdocs[i].ID, Negatives: neg})
					wmu.Unlock()
				}
			}(lo, hi)
		}
		wg.Wait()
		fmt.Printf("  %-10s %6d passages  %.1fs\n", src, n, time.Since(ts).Seconds())
	}
	w.Flush()
	of.Close()
	fmt.Printf("\ndone in %.1fs -> %s\n", time.Since(t0).Seconds(), *outPath)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
