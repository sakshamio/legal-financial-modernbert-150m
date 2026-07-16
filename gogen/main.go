// gogen -- taxonomy-driven synthetic (query -> passage) generation against an OpenAI-compatible LLM
// server (stood up by sparkrun). Pure HTTP + JSON + file IO, no Python libraries, so it belongs in Go
// -- same reasoning as gosample / gopairs / godedup.
//
// WHY GO SPECIFICALLY HERE. The Python version (scripts/gen_synthetic_taxonomy.py, kept as the
// reference) leaked 60GB: ThreadPoolExecutor.map over an unbounded generator queued pending futures
// far faster than the slow API calls drained them. That blowout tripped `earlyoom` (configured
// --prefer python|vllm), which then killed the vLLM server -- the actual cause of every apparent
// "engine crash". A Go worker pool is BOUNDED BY CONSTRUCTION: a fixed set of goroutines each hold
// exactly one in-flight request, so the leak class cannot exist. Memory stays flat, earlyoom stays
// quiet, the server stays up.
//
// Each LLM call returns a JSON cluster {passage, queries[], hard_negative}, yielding len(queries)
// training triples per call. Resumable: counts pairs already on disk and continues.
//
//	go build -o gogen . && ./gogen -target 1000000 -conc 48 -base http://127.0.0.1:8000/v1
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---- taxonomy (mirrors the Python reference) ---------------------------------------------------
var taxonomy = map[string]struct {
	docs    []string
	clauses []string
}{
	"M&A": {
		[]string{"merger agreement", "share purchase agreement", "asset purchase agreement", "letter of intent", "term sheet", "disclosure schedule", "escrow agreement", "transition services agreement", "stockholders' representative agreement"},
		[]string{"representations and warranties", "material adverse change", "purchase price adjustment", "earnout", "indemnification and escrow", "closing conditions", "non-compete and non-solicit", "termination and break fee", "working capital adjustment", "R&W insurance", "covenants between signing and closing", "disclosure schedule exceptions"},
	},
	"Investment Management": {
		[]string{"investment management agreement", "limited partnership agreement", "private placement memorandum", "subscription agreement", "side letter", "sub-advisory agreement", "separately managed account agreement", "fund of funds agreement"},
		[]string{"management fee", "carried interest and distribution waterfall", "high-water mark", "key person provision", "investment restrictions and guidelines", "redemption and lock-up", "most favored nation", "GP clawback", "valuation policy", "capital calls and defaults", "co-investment rights", "fee offset and expense allocation"},
	},
	"Lending & Credit": {
		[]string{"credit agreement", "term loan facility", "revolving credit facility", "security agreement", "guaranty", "intercreditor agreement", "promissory note", "mezzanine note purchase agreement"},
		[]string{"financial covenants", "negative covenants", "events of default", "borrowing base", "affirmative covenants", "mandatory prepayment", "collateral and perfection", "representations and warranties", "conditions precedent", "cash sweep", "make-whole and prepayment premium", "MFN pricing"},
	},
	"NDAs & Confidentiality": {
		[]string{"mutual NDA", "one-way NDA", "M&A due-diligence NDA", "clean team agreement", "data protection addendum"},
		[]string{"definition of confidential information", "permitted disclosures", "residuals clause", "term and survival", "return or destruction of information", "non-solicitation of employees", "standstill", "injunctive relief", "carve-outs from confidentiality"},
	},
	"Fintech & Technology": {
		[]string{"SaaS subscription agreement", "API license agreement", "payment processing agreement", "banking-as-a-service agreement", "data sharing agreement", "embedded finance partnership", "digital asset custody agreement", "master services agreement", "reseller agreement"},
		[]string{"service level agreement", "data security and breach notification", "limitation of liability", "indemnification", "intellectual property ownership", "fees and payment terms", "regulatory compliance and licensing", "termination for convenience", "audit rights", "acceptable use", "uptime and credits", "sub-processor obligations"},
	},
	"Corporate & Securities": {
		[]string{"shareholders agreement", "convertible note", "SAFE agreement", "warrant agreement", "stock option plan", "underwriting agreement", "registration rights agreement", "voting agreement", "prospectus", "offering memorandum"},
		[]string{"liquidation preference", "anti-dilution protection", "drag-along and tag-along", "pre-emptive rights", "board composition", "protective provisions", "conversion mechanics", "information rights", "vesting and acceleration", "lock-up", "use of proceeds", "risk factors"},
	},
	"Derivatives & Structured": {
		[]string{"ISDA master agreement", "credit support annex", "swap confirmation", "repo agreement", "securities lending agreement", "structured note term sheet"},
		[]string{"netting and set-off", "collateral and margin", "events of default and termination", "close-out amount", "eligible collateral and haircuts", "cross-default", "calculation agent", "payment netting", "credit events"},
	},
	"Real Estate & Project Finance": {
		[]string{"commercial lease", "purchase and sale agreement", "commercial real estate loan agreement", "project finance credit agreement", "development agreement", "ground lease"},
		[]string{"rent and escalation", "operating expenses and CAM", "assignment and subletting", "casualty and condemnation", "completion guaranty", "debt service coverage covenant", "reserves and cash management", "permitted transfers", "SNDA"},
	},
	"Regulatory & Compliance": {
		[]string{"Form ADV", "compliance manual", "KYC/AML policy", "code of ethics", "proxy statement", "10-K risk factors section", "MD&A section", "SEC comment letter response"},
		[]string{"conflicts of interest", "custody rule compliance", "best execution", "insider trading policy", "suspicious activity monitoring", "advisory fee disclosure", "material risk factors", "liquidity and capital resources", "critical accounting estimates", "related party transactions"},
	},
	"Asset Management Ops": {
		[]string{"distribution agreement", "transfer agency agreement", "custody agreement", "prime brokerage agreement", "model portfolio agreement", "administration agreement"},
		[]string{"standard of care", "indemnification and liability", "fees and expense reimbursement", "termination and transition", "reporting and recordkeeping", "rehypothecation", "margin and financing", "NAV calculation and error correction", "proxy voting"},
	},
	// ---- expansion: analyst/SEC QnA + structured credit + '40 Act fund docs -------------------
	"Analyst Research & SEC QnA": {
		[]string{"equity research note", "earnings call transcript Q&A", "sell-side initiation report", "credit research note", "MD&A commentary", "investor day presentation", "10-K management discussion", "10-Q results commentary", "8-K event analysis", "guidance revision note", "rating agency commentary"},
		[]string{"revenue drivers and guidance", "gross and operating margin analysis", "segment performance", "free cash flow and capital allocation", "leverage and liquidity", "guidance raise or cut and drivers", "competitive positioning and moat", "valuation rationale and multiples", "near-term catalysts", "KPI and unit economics", "risk factors and headwinds", "management commentary on demand", "backlog and bookings", "working capital and inventory", "capex and reinvestment"},
	},
	"Structured Credit & Securitization": {
		[]string{"ABS offering circular", "RMBS prospectus supplement", "CMBS pooling and servicing agreement", "CLO indenture", "auto loan ABS trust indenture", "credit card master trust agreement", "student loan ABS", "equipment lease ABS", "mortgage servicing agreement", "ABS warehouse facility", "collateral management agreement", "note purchase agreement (securitization)"},
		[]string{"tranching and subordination", "payment waterfall and priority of payments", "credit enhancement and reserve accounts", "overcollateralization test", "interest coverage test", "servicer duties and servicing standard", "eligibility criteria for receivables", "representations and warranties on the pool", "events of default and acceleration", "optional redemption and clean-up call", "reinvestment period and criteria", "collateral quality tests", "excess spread and turbo amortization", "trigger events and early amortization", "risk retention"},
	},
	"Fund Formation & '40 Act": {
		[]string{"CLO collateral management agreement", "1940 Act compliance policy", "closed-end fund charter", "BDC advisory agreement", "interval fund prospectus", "side letter", "LPA amendment", "subscription booklet", "seed investor agreement", "GP commitment letter"},
		[]string{"1940 Act diversification (Subchapter M)", "affiliated transactions (Sections 17(a)/17(d))", "senior securities and leverage limits (Section 18)", "custody rule compliance (17(f))", "fair valuation (Rule 2a-5)", "advisory contract approval (Section 15(c))", "independent director oversight", "CLO reinvestment criteria", "collateral quality and concentration limits", "asset coverage ratio", "co-investment exemptive relief", "most favored nation and side-letter election"},
	},
}

var sectors = []string{"technology", "healthcare", "energy", "financial services", "real estate", "manufacturing", "consumer/retail", "telecommunications", "biotech/pharma", "infrastructure", "media", "renewable energy", "private equity portfolio company", "hedge fund", "insurance"}
var jurisdictions = []string{"Delaware", "New York", "English law", "California", "Cayman Islands", "Luxembourg", "Texas", "Ontario", "Singapore", "Delaware LLC", "Nevada"}
var parties = []string{"a private equity sponsor", "an institutional asset manager", "a growth-stage startup", "a commercial bank", "a family office", "a public company", "a fintech company", "a pension fund", "a sovereign wealth fund", "a hedge fund", "a venture capital firm", "an insurance company", "a REIT", "a broker-dealer"}
var contexts = []string{"a cross-border transaction", "a distressed situation", "a first-time fund", "a syndicated deal", "a bilateral negotiation", "a highly negotiated side letter", "a middle-market deal", "a large-cap transaction", "an amendment and restatement", "a bespoke structure", "a standard-form agreement"}
var queryStyles = []string{
	"a specific factual question a practitioner would ask that this passage answers",
	"a short topical keyword search query (a phrase, not a sentence) about this passage's subject",
	"a natural question phrased the way someone would actually type it into a search box",
}

var categories []string

type cell struct {
	category, subtype, clause, sector, jur, party, ctx, angle, concrete, format string
}

// Real filings/reports are full of TABLES (tranche structures, financial statements, covenant
// compliance, cap tables, fee schedules), and PDF->markdown extraction makes them MESSY -- misaligned
// pipes, merged headers, footnote markers, inconsistent decimals/units. A retriever must handle
// queries over that, so a large share of passages are generated as (or around) a messy markdown
// table. The rest stay prose. ~45% tabular.
var formats = []string{
	"", "", "", "", "", "", // prose (majority weight)
	"Render the passage AS a messy markdown table extracted from a PDF: misaligned pipes, an occasional merged/blank header cell, footnote markers like (1)/(2), and inconsistent decimals or units ($ in thousands vs millions). Put a realistic tranche/financial/fee/covenant table appropriate to the clause, with a short lead-in sentence.",
	"Embed a small, messy markdown table of the key figures inside the passage (columns slightly misaligned, a stray footnote, mixed $mm and $000s), with prose around it.",
	"Include a markdown table with realistic line items relevant to the clause (e.g. tranche/class, balance, coupon, rating; or period, revenue, margin; or fee tier, rate) -- formatted imperfectly as if OCR'd from a filing.",
	"Present a covenant-compliance or capitalization style markdown table (required vs actual, or class vs amount vs %) with minor formatting noise and a footnote.",
}

// Only 815 core (category x doc x clause) cells exist, so at ~1M pairs each is hit ~400 times and the
// model mode-collapses onto canonical phrasings (measured: an opening like "during the term and for a
// period of twelve (12) months following" recurred 14x). To force textual divergence we inject
// randomized CONCRETE PARTICULARS -- specific figures/dates/names the passage must incorporate -- plus
// a random stylistic angle. Concrete specifics make even same-clause passages read as distinct
// instances rather than templates.
var angles = []string{
	"heavily negotiated with unusual carve-outs", "borrower/obligor-favorable", "counterparty-favorable",
	"a short, tightly drafted version", "a long, comprehensive version with sub-clauses",
	"with an atypical exception or proviso", "cross-referencing several defined terms and schedules",
	"plain-language / modern-drafting style", "traditional/formal drafting style",
	"with specific numeric thresholds and triggers", "amended-and-restated with a conforming change",
}

// Invented entity stems -- for fictional counterparties, portfolio companies, SPVs, and issuers.
var entityStems = []string{"Aldermere", "Brightwater", "Calderon", "Deverell", "Ellingham", "Fairmont Ridge",
	"Granville", "Harnwell", "Ironbridge", "Juniper Peak", "Kestrelton", "Larkspur", "Montclair Systems",
	"Northgate", "Orrington", "Pemberton", "Quillfield", "Rosseland", "Sterling Vale", "Thornbury",
	"Umberland", "Vanterra", "Westmark", "Yarborough", "Zephyr Cove", "Ashford Bay", "Coldwater",
	"Dunmore", "Everly", "Hollingsworth", "Marlowe", "Redfern", "Stanhope", "Wexford"}

// Real firms across banks, PE/credit funds, and asset managers -- structured credit, IMAs, and side
// letters really involve these players, so naming them makes retrieval queries realistic. Used only
// as training signal inside clearly-synthetic passages (see the dataset card).
var realFirms = []string{"JPMorgan", "Goldman Sachs", "Morgan Stanley", "Bank of America", "Citigroup",
	"Wells Fargo", "Deutsche Bank", "Barclays", "BNP Paribas", "MUFG",
	"Blackstone", "Apollo Global", "KKR", "Carlyle Group", "Ares Management", "Brookfield", "Oaktree",
	"Blue Owl", "Sixth Street", "HPS Investment Partners", "Golub Capital", "Antares Capital",
	"Angelo Gordon", "Monroe Capital", "Diameter Capital", "Sculptor Capital",
	"BlackRock", "PIMCO", "Fidelity", "Vanguard", "State Street", "Nuveen", "Invesco", "Franklin Templeton",
	"Berkshire Hathaway", "Prudential", "MetLife",
	// niche / mid-market names
	"Varagon Capital", "Twin Brook Capital", "Comvest Partners", "MidCap Financial", "Churchill Asset Management",
	"Benefit Street Partners", "Crescent Capital", "Audax Group", "NewStar Financial", "Fortress"}

func pickEntity(r *rand.Rand, suffix string) string {
	if r.Intn(100) < 55 { // ~55% real firm, ~45% invented counterparty/SPV
		return realFirms[r.Intn(len(realFirms))]
	}
	return entityStems[r.Intn(len(entityStems))] + suffix
}

func randConcrete(r *rand.Rand) string {
	pct := 1 + r.Intn(1499) // 0.01%..15.00%
	bps := []int{25, 50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 500, 650}[r.Intn(13)]
	amt := (1 + r.Intn(4990)) * 100000 // $100k..$499M
	months := []int{3, 6, 12, 18, 24, 36, 48, 60, 84}[r.Intn(9)]
	days := []int{5, 10, 15, 30, 45, 60, 90}[r.Intn(7)]
	ratio := []string{"1.10x", "1.25x", "1.50x", "2.00x", "2.50x", "3.00x", "3.50x", "4.25x", "5.00x", "6.00x"}[r.Intn(10)]
	rating := []string{"AAA/Aaa", "AA/Aa2", "A/A2", "BBB/Baa2", "BB/Ba2", "B/B2"}[r.Intn(6)]
	e1 := pickEntity(r, " Holdings")
	e2 := pickEntity(r, " Capital")
	yr := 2018 + r.Intn(8)
	q := []string{"Q1", "Q2", "Q3", "Q4"}[r.Intn(4)]
	return fmt.Sprintf("weave in these specific particulars (invent more as needed) and keep the numbers realistic: a party named %q and a counterparty named %q; a principal/notional around $%s; a coupon/rate near %.2f%% (or +%d bps over SOFR); a coverage/leverage ratio of %s; a %s tranche/rating; a period of %d months; a %d-day notice/cure; a reporting period of %s %d",
		e1, e2, commas(amt), float64(pct)/100.0, bps, ratio, rating, months, days, q, yr)
}

func commas(n int) string {
	s := fmt.Sprintf("%d", n)
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	return string(out)
}

func sampleCell(r *rand.Rand) cell {
	cat := categories[r.Intn(len(categories))]
	t := taxonomy[cat]
	return cell{cat, t.docs[r.Intn(len(t.docs))], t.clauses[r.Intn(len(t.clauses))],
		sectors[r.Intn(len(sectors))], jurisdictions[r.Intn(len(jurisdictions))],
		parties[r.Intn(len(parties))], contexts[r.Intn(len(contexts))],
		angles[r.Intn(len(angles))], randConcrete(r), formats[r.Intn(len(formats))]}
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func buildMessages(c cell, nq int) []chatMsg {
	// The JSON ENVELOPE has no code fences, but the "passage"/"hard_negative" string VALUES may contain
	// markdown tables (newlines and pipes, properly JSON-escaped). That is intended, not a violation.
	sys := "You generate realistic financial/legal training data. Respond with a single valid JSON object and nothing else -- no code fences around the JSON, no commentary. The passage string itself MAY contain a markdown table (escape newlines as \\n)."
	styles := strings.Join(queryStyles[:nq], "; ")
	user := fmt.Sprintf(`You are an expert transactional attorney and financial analyst. Produce one realistic training cluster for a retrieval model.

Setting (use it to make the text specific and varied):
- Document category: %s
- Document type: %s
- Clause / topic focus: %s
- Sector: %s
- Governing law: %s
- A principal party: %s
- Context: %s
- Drafting angle: %s
- To make this a DISTINCT instance rather than a generic template, %s
- Format: %s

Return a JSON object with exactly these keys:
- "passage": a realistic 90-180 word excerpt from the "%s" focused on "%s". Write it the way a real %s reads -- defined terms, cross-references, appropriate legal/financial register, reflecting the drafting angle, the specific particulars, and the Format instruction above. Do NOT include a heading or the document title; just the operative text (with a table if the Format asks for one). If a table is included, at least one query MUST ask about a specific value in it.
- "queries": a list of %d DISTINCT search queries that this passage answers. Vary them: (%s). Do not copy long phrases from the passage verbatim.
- "hard_negative": a realistic excerpt that is SIMILAR in topic/document type (a related clause or a neighbouring provision, matching the Format) but that does NOT actually answer the queries -- a plausible wrong retrieval result.

Output only the JSON object.`,
		c.category, c.subtype, c.clause, c.sector, c.jur, c.party, c.ctx, c.angle, c.concrete,
		fmtOrProse(c.format), c.subtype, c.clause, c.subtype, nq, styles)
	return []chatMsg{{"system", sys}, {"user", user}}
}

func fmtOrProse(f string) string {
	if f == "" {
		return "plain prose (no table)"
	}
	return f
}

type cluster struct {
	Passage      string   `json:"passage"`
	Queries      []string `json:"queries"`
	HardNegative string   `json:"hard_negative"`
}

// parseCluster extracts the JSON object from the model output, tolerant of code fences / stray prose.
func parseCluster(text string) *cluster {
	text = strings.TrimSpace(text)
	if strings.Contains(text, "```") {
		parts := strings.Split(text, "```")
		if len(parts) >= 2 {
			text = strings.TrimPrefix(strings.TrimSpace(parts[1]), "json")
		}
	}
	a := strings.Index(text, "{")
	b := strings.LastIndex(text, "}")
	if a < 0 || b <= a {
		return nil
	}
	var cl cluster
	if json.Unmarshal([]byte(text[a:b+1]), &cl) != nil {
		return nil
	}
	kept := cl.Queries[:0]
	for _, q := range cl.Queries {
		q = strings.TrimSpace(q)
		if len(q) >= 8 && len(q) <= 300 {
			kept = append(kept, q)
		}
	}
	cl.Queries = kept
	if len(cl.Passage) < 120 || len(cl.Queries) == 0 {
		return nil
	}
	if len(cl.HardNegative) < 120 {
		cl.HardNegative = ""
	}
	return &cl
}

type outRec struct {
	Anchor   string            `json:"anchor"`
	Positive string            `json:"positive"`
	Label    string            `json:"label"`
	Source   string            `json:"source"`
	Meta     map[string]string `json:"meta"`
	Negative string            `json:"negative_0,omitempty"`
}

func main() {
	target := flag.Int("target", 1000000, "pair target")
	nq := flag.Int("nq", 3, "queries per cluster")
	base := flag.String("base", "http://127.0.0.1:8000/v1", "OpenAI-compatible endpoint")
	conc := flag.Int("conc", 48, "concurrent requests (fixed goroutines -> bounded memory)")
	maxTok := flag.Int("max-tokens", 700, "")
	seed := flag.Int64("seed", 0, "")
	out := flag.String("out", "../data/pairs_synth_taxonomy", "")
	flag.Parse()

	for k := range taxonomy {
		categories = append(categories, k)
	}

	// discover model id
	modelID := discoverModel(*base)
	if modelID == "" {
		fmt.Fprintln(os.Stderr, "no server / model at", *base)
		os.Exit(1)
	}

	os.MkdirAll(*out, 0o755)
	outPath := filepath.Join(*out, "train.jsonl")
	progPath := filepath.Join(*out, "progress.json")

	// resume: count existing pairs
	have := countLines(outPath)
	var pairs int64 = int64(have)
	var calls, fails int64
	fmt.Printf("server %s | model %s | have %d pairs, target %d\n", *base, modelID, have, *target)

	f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		panic(err)
	}
	w := bufio.NewWriterSize(f, 1<<20)
	var wmu sync.Mutex

	client := &http.Client{Timeout: 180 * time.Second}
	t0 := time.Now()
	var wg sync.WaitGroup
	stop := make(chan struct{})

	// FIXED pool of goroutines. Each holds exactly one in-flight request -> memory is O(conc), never
	// unbounded. This is the whole point of the Go rewrite.
	for i := 0; i < *conc; i++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(*seed + int64(worker)*7919 + int64(have)))
			for {
				select {
				case <-stop:
					return
				default:
				}
				if atomic.LoadInt64(&pairs) >= int64(*target) {
					return
				}
				c := sampleCell(r)
				cl := generate(client, *base, modelID, buildMessages(c, *nq), *maxTok)
				n := atomic.AddInt64(&calls, 1)
				if cl == nil {
					atomic.AddInt64(&fails, 1)
				} else {
					lab := fmt.Sprintf("synth-%s-%s-%d", c.category, c.subtype, n)
					src := "synth_" + strings.ToLower(strings.Fields(c.category)[0])
					meta := map[string]string{"category": c.category, "subtype": c.subtype, "clause": c.clause}
					wmu.Lock()
					for _, q := range cl.Queries {
						rec := outRec{"[QUERY] " + q, "[PASSAGE] " + cl.Passage, lab, src, meta, ""}
						if cl.HardNegative != "" {
							rec.Negative = "[PASSAGE] " + cl.HardNegative
						}
						b, _ := json.Marshal(rec)
						w.Write(b)
						w.WriteByte('\n')
						atomic.AddInt64(&pairs, 1)
					}
					wmu.Unlock()
				}
				if n%25 == 0 { // flush often so the streaming uploader sees fresh data
					wmu.Lock()
					w.Flush()
					wmu.Unlock()
				}
				if n%200 == 0 {
					el := time.Since(t0).Seconds()
					p := atomic.LoadInt64(&pairs)
					pps := float64(p-int64(have)) / el
					eta := float64(int64(*target)-p) / (pps + 1e-9) / 3600
					wmu.Lock()
					w.Flush()
					os.WriteFile(progPath, []byte(fmt.Sprintf(`{"pairs":%d,"calls":%d,"fail_rate":%.3f}`,
						p, n, float64(atomic.LoadInt64(&fails))/float64(n))), 0o644)
					wmu.Unlock()
					fmt.Printf("  pairs %d/%d | %.0f pairs/s | fail %.1f%% | ETA %.1fh\n",
						p, *target, pps, 100*float64(atomic.LoadInt64(&fails))/float64(n), eta)
				}
			}
		}(i)
	}
	wg.Wait()
	w.Flush()
	f.Close()
	fmt.Printf("\ndone: %d pairs in %.1fh (fail %.1f%%)\n", atomic.LoadInt64(&pairs),
		time.Since(t0).Hours(), 100*float64(fails)/float64(calls+1))
}

func discoverModel(base string) string {
	resp, err := http.Get(base + "/models")
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var m struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.NewDecoder(resp.Body).Decode(&m) != nil || len(m.Data) == 0 {
		return ""
	}
	return m.Data[0].ID
}

func generate(client *http.Client, base, model string, msgs []chatMsg, maxTok int) *cluster {
	body, _ := json.Marshal(map[string]interface{}{
		"model": model, "messages": msgs, "max_tokens": maxTok,
		"temperature": 0.9, "top_p": 0.95,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	})
	resp, err := client.Post(base+"/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		io.Copy(io.Discard, resp.Body)
		return nil
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.NewDecoder(resp.Body).Decode(&out) != nil || len(out.Choices) == 0 {
		return nil
	}
	c := out.Choices[0].Message.Content
	cl := parseCluster(c)
	if cl == nil && os.Getenv("GOGEN_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "PARSE-FAIL len=%d content=%q\n", len(c), c[:min(len(c), 300)])
	}
	return cl
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func countLines(path string) int {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	n := 0
	sc := bufio.NewReaderSize(f, 1<<20)
	for {
		_, err := sc.ReadBytes('\n')
		if err != nil {
			break
		}
		n++
	}
	return n
}
