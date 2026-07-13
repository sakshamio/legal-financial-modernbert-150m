// gosample -- sample text chunks from the deduped corpus for geometry distillation.
//
// Port of scripts/sample_distill_corpus.py, which is kept as the reference implementation. Unlike
// godedup, the two CANNOT be byte-identical -- they draw offsets from different RNGs, so they see
// different windows. What must match is the extraction logic, checked distributionally: reject rate,
// length distribution, domain mixture, and zero corruption (see the smoke test in the commit).
//
// Pure text processing, no Python libraries needed, so it belongs in Go -- same reasoning as godedup,
// which ran 24x faster than its Python reference.
//
// BOUNDED-WINDOW SAMPLING
// The obvious approach -- seek to a random offset, read the whole line, parse the JSON, take a chunk
// -- is a trap on this corpus. Record sizes are wildly skewed:
//
//	legal_caselaw       median record    15 KB
//	financial_edgar     median record   179 KB
//	legal_regulations   median record  3889 KB   (mean 10 MB)
//
// so it reads and parses a 10 MB document to keep 1,800 characters: 0.02% of the bytes it moved.
// Instead we read a small bounded window at the offset and recover the text from inside the JSON
// string without parsing the enclosing record. Records are {"text":"...","source":...,"id":...} with
// `text` first, so a window landing mid-record is almost entirely escaped text; the only structure
// we must respect is the first UNESCAPED quote, which ends the value. Cost becomes O(chunks),
// independent of document size.
//
// Sampling uniformly BY BYTE also weights documents by length, which matches the token distribution
// the encoder was actually pretrained on. One-chunk-per-document would over-sample short documents.
//
// MEMORY
// Nothing is accumulated. Workers stream to per-domain shards; the merge shuffles an array of domain
// LABELS (one byte per chunk, ~40MB at the 40M rung) rather than the text (~72GB), then interleaves
// the shards. An OOM already killed corpus prep once on this box; the 40M rung would not fit.
//
//	go build -o gosample . && ./gosample -n 40000000 -out ../data/distill
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Same mixture as the pretraining corpus (fractions of deduped bytes).
var mixture = []struct {
	name string
	frac float64
}{
	{"financial_edgar", 0.24},
	{"financial_pol", 0.07},
	{"general_web", 0.24},
	{"legal_caselaw", 0.29},
	{"legal_contracts", 0.12},
	{"legal_regulations", 0.04},
}

type rec struct {
	Text   string `json:"text"`
	Domain string `json:"domain"`
}

// firstUnescapedQuote returns the index of the first `"` not preceded by an odd run of backslashes.
func firstUnescapedQuote(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] != '"' {
			continue
		}
		b := 0
		for j := i - 1; j >= 0 && s[j] == '\\'; j-- {
			b++
		}
		if b%2 == 0 {
			return i
		}
	}
	return -1
}

// unescape decodes a fragment of a JSON string value, trimming partial escapes at either edge.
func unescape(frag string) (string, bool) {
	for a := 0; a < 3; a++ {
		for b := 0; b < 3; b++ {
			if a+b >= len(frag) {
				continue
			}
			var out string
			if err := json.Unmarshal([]byte(`"`+frag[a:len(frag)-b]+`"`), &out); err == nil {
				return out, true
			}
		}
	}
	return "", false
}

// isControl matches Python's unicodedata.category(c) == "Cc", i.e. BOTH the C0 range and the C1
// range (0x80-0x9f). Omitting C1 let 336 control chars per 60k chunks through.
func isControl(r rune) bool {
	if r == '\n' || r == '\t' || r == '\r' {
		return false
	}
	return r < 0x20 || (r >= 0x7f && r <= 0x9f)
}

// truncRunes cuts to at most n BYTES without splitting a rune. Slicing text[:n] directly severs
// multi-byte UTF-8 characters, and the JSON encoder then writes the debris as U+FFFD -- corruption
// introduced by us, not present in the corpus.
func truncRunes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// hasTornEscape reports a leftover backslash escape, i.e. the window edge cut one in half.
func hasTornEscape(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '\\' && strings.IndexByte("nrtu\"", s[i+1]) >= 0 {
			return true
		}
	}
	return false
}

type result struct {
	domain string
	n      int
	misses int
}

func sampleFile(dedupDir, outDir, domain string, nWant, minChars, maxChars int, seed int64) result {
	path := filepath.Join(dedupDir, domain+".jsonl")
	fi, err := os.Stat(path)
	if err != nil {
		fmt.Printf("  !! missing %s\n", domain)
		return result{domain, 0, 0}
	}
	size := fi.Size()

	f, err := os.Open(path)
	if err != nil {
		return result{domain, 0, 0}
	}
	defer f.Close()

	shard, err := os.Create(filepath.Join(outDir, "_shard_"+domain+".jsonl"))
	if err != nil {
		return result{domain, 0, 0}
	}
	w := bufio.NewWriterSize(shard, 4<<20)
	defer func() { w.Flush(); shard.Close() }()

	rng := rand.New(rand.NewSource(seed))
	win := maxChars * 3 // escapes expand; 3x gives ample slack to land maxChars of real text
	buf := make([]byte, win)
	seen := make(map[uint64]struct{}, nWant)
	enc := json.NewEncoder(w)

	nOut, misses, maxMiss := 0, 0, nWant*10+500
	for nOut < nWant && misses < maxMiss {
		off := rng.Int63n(max64(1, size-int64(win)))
		n, _ := f.ReadAt(buf, off)
		if n == 0 {
			misses++
			continue
		}
		s := string(buf[:n])
		if !utf8.ValidString(s) {
			s = strings.ToValidUTF8(s, "") // drop torn utf-8 at the window edges
		}

		// if we landed on a record start, skip past the key into the value
		if k := strings.Index(s, `{"text":"`); k >= 0 {
			s = s[k+9:]
		}
		// the value ends at the first unescaped quote (then comes ,"source":...)
		if q := firstUnescapedQuote(s); q >= 0 {
			s = s[:q]
		}

		text, ok := unescape(s)
		if !ok || len(text) < minChars || hasTornEscape(text) {
			misses++
			continue
		}
		text = strings.Map(func(r rune) rune {
			if isControl(r) || r == utf8.RuneError {
				return -1
			}
			return r
		}, text)

		// snap to whitespace so the teacher never sees a half-word at either end.
		// Every cut here is on a space or a rune boundary, so none of them can split a character.
		text = truncRunes(text, maxChars)
		if c := strings.IndexByte(text, ' '); c >= 0 && c < 40 {
			text = text[c+1:]
		}
		if c := strings.LastIndexByte(text, ' '); c > len(text)-40 {
			text = text[:c]
		}
		text = strings.TrimSpace(text)
		if len(text) < minChars {
			misses++
			continue
		}
		// belt and braces: never hand the teacher a chunk we corrupted
		if strings.ContainsRune(text, utf8.RuneError) {
			misses++
			continue
		}

		h := fnv64(text)
		if _, dup := seen[h]; dup {
			misses++
			continue
		}
		seen[h] = struct{}{}
		if err := enc.Encode(rec{Text: text, Domain: domain}); err != nil {
			break
		}
		nOut++
	}
	return result{domain, nOut, misses}
}

// fnv64 over the first 512 bytes -- the dedup key. 40M entries at 8 bytes stays ~1GB in a Go map;
// a Python set of the same hashes would be several times that.
func fnv64(s string) uint64 {
	if len(s) > 512 {
		s = s[:512]
	}
	var h uint64 = 14695981039346656037
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func main() {
	var (
		n        = flag.Int("n", 5_000_000, "train chunks")
		valN     = flag.Int("val", 4096, "held-out chunks (taken from the FRONT of the shuffled stream)")
		minChars = flag.Int("min-chars", 400, "")
		maxChars = flag.Int("max-chars", 1800, "~450 teacher tokens")
		seed     = flag.Int64("seed", 0, "")
		dedupDir = flag.String("dedup", "../data/raw_dedup", "")
		outDir   = flag.String("out", "../data/distill", "")
	)
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		panic(err)
	}
	total := *n + *valN
	t0 := time.Now()

	// one goroutine per domain file; the 10MB-record file is the long pole
	var wg sync.WaitGroup
	var mu sync.Mutex
	counts := map[string]int{}
	for i, m := range mixture {
		wg.Add(1)
		go func(i int, name string, frac float64) {
			defer wg.Done()
			r := sampleFile(*dedupDir, *outDir, name, int(float64(total)*frac), *minChars, *maxChars,
				*seed+int64(i))
			mu.Lock()
			counts[r.domain] = r.n
			fmt.Printf("  %-20s %10d chunks  (%d rejects)  [%.1f min]\n",
				r.domain, r.n, r.misses, time.Since(t0).Minutes())
			mu.Unlock()
		}(i, m.name, m.frac)
	}
	wg.Wait()

	// --- shuffled merge, streaming.
	// Shuffle an array of DOMAIN LABELS (one byte per chunk), not the text. Each shard is already in
	// random corpus order because it was built from random byte offsets, so emitting shards in a
	// shuffled label order yields a properly shuffled stream with ~nothing resident.
	var doms []string
	for d, c := range counts {
		if c > 0 {
			doms = append(doms, d)
		}
	}
	sort.Strings(doms) // deterministic label->shard mapping for a given seed

	order := make([]uint8, 0, total)
	for i, d := range doms {
		for j := 0; j < counts[d]; j++ {
			order = append(order, uint8(i))
		}
	}
	rng := rand.New(rand.NewSource(*seed))
	rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
	fmt.Printf("\n  merging %d chunks (label array %.0f MB)\n", len(order), float64(len(order))/1e6)

	readers := make([]*bufio.Reader, len(doms))
	files := make([]*os.File, len(doms))
	for i, d := range doms {
		f, err := os.Open(filepath.Join(*outDir, "_shard_"+d+".jsonl"))
		if err != nil {
			panic(err)
		}
		files[i] = f
		readers[i] = bufio.NewReaderSize(f, 4<<20)
	}

	trF, _ := os.Create(filepath.Join(*outDir, "train.jsonl"))
	vaF, _ := os.Create(filepath.Join(*outDir, "val.jsonl"))
	trW := bufio.NewWriterSize(trF, 8<<20)
	vaW := bufio.NewWriterSize(vaF, 8<<20)

	written := map[string]int{"train": 0, "val": 0}
	for k, i := range order {
		// ReadBytes, never bufio.Scanner: a Scanner silently returns false on a line over its buffer
		// and reports SUCCESS. That exact bug truncated legal_regulations to 36 of 88,804 documents
		// during dedup and reported success.
		line, err := readers[i].ReadBytes('\n')
		if len(line) == 0 || err != nil {
			continue
		}
		// val is the FRONT of the shuffled stream, so it is disjoint from every prefix rung
		if k < *valN {
			vaW.Write(line)
			written["val"]++
		} else {
			trW.Write(line)
			written["train"]++
		}
	}
	trW.Flush()
	vaW.Flush()
	trF.Close()
	vaF.Close()
	for i, f := range files {
		f.Close()
		os.Remove(filepath.Join(*outDir, "_shard_"+doms[i]+".jsonl"))
	}

	for _, name := range []string{"train", "val"} {
		fi, _ := os.Stat(filepath.Join(*outDir, name+".jsonl"))
		fmt.Printf("  %-6s %10d chunks  %6.1f GB\n", name, written[name], float64(fi.Size())/1e9)
	}
	el := time.Since(t0).Seconds()
	fmt.Printf("\ndone in %.1f min  (%.0f chunks/s)\n", el/60,
		float64(written["train"]+written["val"])/el)
}
