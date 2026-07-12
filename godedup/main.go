// Parallel near-dedup of the pretraining corpus.
//
// The Python version ran at ~25 MB/s on ONE of 20 cores (~87 min for 130GB). Dedup looks inherently
// sequential -- "keep first occurrence" needs a global order -- but only the cheap set lookup actually
// needs ordering. The expensive work (JSON parse, paragraph split, normalize, hash) is per-paragraph
// and embarrassingly parallel. So: workers hash, a single owner goroutine decides keep/drop in strict
// document order, and correctness is identical to the sequential version.
//
// Also 3x more memory-compact than Python: map[uint64]struct{} vs a Python set of int objects, which
// matters at ~260M paragraphs.
//
// Normalization (lowercase, digits->#, strip punctuation) is what makes this a NEAR-dedup: it catches
// boilerplate differing only by date, dollar amount, or section number -- which is most of it.
// Measured: EDGAR 12.3% duplicate paragraphs, contracts 10.8%, case law 0.6%.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"hash/maphash"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unicode"
)

const (
	minParaChars  = 120 // shorter fragments (headings, "Not Applicable") repeat legitimately
	minDocChars   = 200
	batchSize     = 256
	maxBatchBytes = 32 << 20 // cap batches by bytes too: one 168MB doc must not blow a worker
)

// Sources in dedup priority order: first occurrence wins, so the highest-signal corpus keeps the copy.
// general_web is LAST -- if a legal passage also appears on the web, we keep the legal one.
var sources = []string{
	"legal_caselaw",
	"legal_regulations",
	"legal_contracts",
	"financial_edgar",
	"financial_pol",
	"general_web",
}

type record struct {
	Text   string `json:"text"`
	Source string `json:"source"`
	ID     string `json:"id"`
}

// para is one paragraph: its original text, plus the hash of its normalized form.
// hashed=false means it's too short to dedup -- always kept.
type para struct {
	text   string
	h      uint64
	hashed bool
}

type doc struct {
	id    string
	paras []para
}

var seed = maphash.MakeSeed()

// normalize folds away the things boilerplate varies by: case, digits (dates/amounts/section numbers),
// punctuation, and whitespace runs.
func normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := false
	for _, r := range s {
		switch {
		case unicode.IsDigit(r):
			b.WriteByte('#')
			prevSpace = false
		case unicode.IsSpace(r):
			if !prevSpace {
				b.WriteByte(' ')
			}
			prevSpace = true
		case unicode.IsLetter(r):
			b.WriteRune(unicode.ToLower(r))
			prevSpace = false
		default: // drop punctuation
		}
	}
	return b.String()
}

func collapseWS(s string) string { return strings.Join(strings.Fields(s), " ") }

// splitParas mirrors Python's re.split(r"\n\s*\n"): blank-line separated blocks.
func splitParas(text string) []string {
	lines := strings.Split(text, "\n")
	var out []string
	var cur []string
	for _, ln := range lines {
		if strings.TrimSpace(ln) == "" {
			if len(cur) > 0 {
				out = append(out, strings.Join(cur, "\n"))
				cur = cur[:0]
			}
		} else {
			cur = append(cur, ln)
		}
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, "\n"))
	}
	return out
}

func processBatch(lines [][]byte) []doc {
	docs := make([]doc, 0, len(lines))
	for _, ln := range lines {
		var r record
		if err := json.Unmarshal(ln, &r); err != nil {
			continue
		}
		d := doc{id: r.ID}
		for _, p := range splitParas(r.Text) {
			c := collapseWS(p)
			if c == "" {
				continue
			}
			if len(c) < minParaChars {
				d.paras = append(d.paras, para{text: p, hashed: false})
				continue
			}
			d.paras = append(d.paras, para{text: p, h: maphash.String(seed, normalize(c)), hashed: true})
		}
		docs = append(docs, d)
	}
	return docs
}

func dedupSource(src string, rawDir, outDir string, seen map[uint64]struct{}) (kept, dropped int64, dIn, dOut int) {
	in, err := os.Open(filepath.Join(rawDir, src+".jsonl"))
	if err != nil {
		fmt.Printf("[%s] MISSING -- skipped\n", src)
		return
	}
	defer in.Close()
	out, err := os.Create(filepath.Join(outDir, src+".jsonl"))
	if err != nil {
		panic(err)
	}
	defer out.Close()

	nw := runtime.NumCPU() - 2
	if nw < 1 {
		nw = 1
	}
	// Fan out batches to workers; each worker returns results on its OWN channel, and we read those
	// channels round-robin. That preserves strict global document order (required for first-occurrence
	// semantics) while still hashing in parallel.
	type job struct {
		lines [][]byte
		out   chan []doc
	}
	jobs := make(chan job, nw*2)
	var wg sync.WaitGroup
	for i := 0; i < nw; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				j.out <- processBatch(j.lines)
			}
		}()
	}

	results := make(chan chan []doc, nw*2)
	go func() {
		// bufio.Reader.ReadBytes, NOT bufio.Scanner. Scanner has a max token size and on a longer line
		// it silently returns false with no error -- it does not fail, it just STOPS. legal_regulations
		// contains a 168MB single line (state codes are enormous), which truncated that source to 36 of
		// 88,804 documents and reported success. Silent 99.96% data loss. ReadBytes is unbounded.
		rd := bufio.NewReaderSize(in, 1<<20)
		batch := make([][]byte, 0, batchSize)
		batchBytes := 0
		flush := func() {
			if len(batch) == 0 {
				return
			}
			ch := make(chan []doc, 1)
			jobs <- job{lines: batch, out: ch}
			results <- ch
			batch = make([][]byte, 0, batchSize)
			batchBytes = 0
		}
		for {
			line, err := rd.ReadBytes('\n')
			if len(line) > 0 {
				batch = append(batch, line)
				batchBytes += len(line)
				// Cap batches by BYTES as well as count: 256 x 168MB documents would blow worker memory.
				if len(batch) >= batchSize || batchBytes >= maxBatchBytes {
					flush()
				}
			}
			if err != nil {
				break // io.EOF, or a real read error -- either way we are done with this file
			}
		}
		flush()
		close(jobs)
		wg.Wait()
		close(results)
	}()

	w := bufio.NewWriterSize(out, 8<<20)
	defer w.Flush()
	enc := json.NewEncoder(w)

	for ch := range results { // strict order
		for _, d := range <-ch {
			dIn++
			var keep []string
			for _, p := range d.paras {
				if !p.hashed {
					keep = append(keep, p.text)
					kept += int64(len(p.text))
					continue
				}
				if _, dup := seen[p.h]; dup {
					dropped += int64(len(p.text))
					continue
				}
				seen[p.h] = struct{}{}
				keep = append(keep, p.text)
				kept += int64(len(p.text))
			}
			text := strings.TrimSpace(strings.Join(keep, "\n\n"))
			if len(text) < minDocChars {
				continue
			}
			dOut++
			if err := enc.Encode(record{Text: text, Source: src, ID: d.id}); err != nil {
				panic(err)
			}
		}
	}
	return
}

func main() {
	root, _ := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), ".."))
	if len(os.Args) > 1 {
		root = os.Args[1]
	}
	rawDir := filepath.Join(root, "data", "raw")
	outDir := filepath.Join(root, "data", "raw_dedup")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}

	fmt.Printf("godedup: %d cores, raw=%s\n\n", runtime.NumCPU(), rawDir)
	seen := make(map[uint64]struct{}, 1<<28)

	var totKept, totDropped int64
	for _, src := range sources {
		k, d, dIn, dOut := dedupSource(src, rawDir, outDir, seen)
		if dIn == 0 {
			continue
		}
		totKept += k
		totDropped += d
		pct := 100 * float64(d) / float64(k+d)
		fmt.Printf("%-20s docs %8d -> %8d | dropped %6.2f GB (%5.1f%%) | kept %6.2f GB\n",
			src, dIn, dOut, float64(d)/1e9, pct, float64(k)/1e9)
	}

	fmt.Printf("\n=== summary ===\n")
	fmt.Printf("unique paragraphs hashed: %d\n", len(seen))
	fmt.Printf("dropped %.2f GB of duplicated text (%.1f%% of corpus)\n",
		float64(totDropped)/1e9, 100*float64(totDropped)/float64(totKept+totDropped))
	fmt.Printf("corpus after dedup: %.2f GB\n", float64(totKept)/1e9)
}
