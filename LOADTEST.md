# Load test and live verification

Everything below was measured on one developer laptop (macOS, Python 3.12) that was **also running
Chrome, VS Code helpers and other agents**: `uptime` showed a load average between 10 and 310 during
these runs. Absolute numbers are therefore pessimistic and noisy; the *relative* before/after
changes and the identified bottlenecks are what to trust. The database was the real MongoDB Atlas
free-tier (M0) cluster from `.env`. The LLM providers were **fakes** (no spend) except in the live E2E.

## 1. Setup and how to run

```bash
# prerequisites: the project venv (uvicorn, httpx, pymongo, pypdf ...) and MONGODB_URI in .env
python scripts/loadtest.py --scenarios a,b,c,d,e --users 200 --json out.json
python scripts/loadtest.py --scenarios b --users 50        # smaller / quicker
python scripts/loadtest.py --cleanup-only                  # idempotent: removes every loadtest_ doc
python scripts/fake_providers.py --port 8890 --anthropic-fail 0.5   # run the fake providers by hand
pytest tests/test_loadtest_support.py                      # unit tests for the hook, fakes, PDF cache
```

* `scripts/fake_providers.py` is an async Starlette app that speaks the Anthropic (`/v1/messages`),
  Groq (`/openai/v1/chat/completions`) and Gemini (`:generateContent`, `:streamGenerateContent`,
  plus `:embedContent` / `:batchEmbedContents`) wire formats, JSON and SSE, with configurable latency,
  429/503/529 injection rates and hangs. Replies are deterministic and shaped for what each feature
  parses (quiz, flashcards, mind map, exam, teach-back, insights, overview, intent/rewrite/grade).
* `backend/llm_chain.py` and `backend/gemini_embeddings.py` honour **`LLM_CHAIN_TEST_BASE_URL`**
  (test only). It is ignored whenever `VERCEL` is set, so it can never redirect production traffic or
  keys. `loadtest.py` sets it for the server it spawns; real provider hosts are never contacted.
* `scripts/loadtest.py` spawns the fake providers and `uvicorn api.index:app`, provisions throwaway
  guests `guest_loadtest001..` (hard cap 250) straight in MongoDB (`POST /api/guest` is throttled to
  15 per IP per 10 min per instance), samples server RSS at 1 Hz, probes `/api/health` at 10 Hz
  (event-loop blocking shows up as health p99) and measures its own loop lag. On exit it deletes
  everything keyed by the `guest_loadtest` owner prefix: artifacts, guests, GridFS uploads and index
  blobs, and local `documents/user__guest_loadtest*` folders. Verified idempotent
  (`--cleanup-only` afterwards reports nothing left).

Scenarios: **A** health / config / subjects / static shell at 200 concurrent connections (10 s each);
**B** 200 concurrent users, each its own guest: list subjects, history, open chat, stream 3 messages
with ~2 s think time, quiz (3 q), flashcards (3), history, progress; **C** = B with Anthropic 50% HTTP
529 and Groq 30% HTTP 429; **D** = B on `uvicorn --workers 4`; **E** 8 concurrent uploads of a 1.5 MB PDF.

> The per-owner rate limiter (`backend/ratelimit.py`) would throttle the many guests this script creates.
> `scripts/loadtest.py` starts its own server with `RATE_LIMIT_DISABLED=1`; set the same variable if you point
> it at a server you started yourself. The flag is ignored when `VERCEL` is set, so a deployed site is always limited.

## 2. Results

### Scenario B (200 users, 1 worker): before and after

| | original code | + PDF-text cache, bigger thread pools | + pool wait 10 s, resilient history save (final) |
|---|---|---|---|
| wall time for 2000 requests | 314 s | 68.7 s | 70.2 s |
| 5xx | 1 | 15 (Mongo pool wait 5 s) | **0** |
| client read timeouts (90 s) | 305 | 2 (conn reset) | **0** |
| `POST /api/quiz/generate` p50 / p95 | 90 s (timeout) / 92 s | 2.7 s / 5.9 s | 2.9 s / 5.7 s |
| `POST /api/flashcards/generate` p50 / p95 | 90 s (timeout) / 90 s | 3.3 s / 8.5 s | 3.2 s / 5.9 s |
| chat stream TTFT p50 / p95 | 9.7 s / 13.2 s | 2.75 s / 6.4 s | 2.9 s / 6.7 s |
| chat stream total p50 / p95 | 13.7 s / 21.6 s | 5.9 s / 11.3 s | 6.2 s / 12.3 s |
| `GET /api/history/quizzes` p50 | 45 s | 2.8 s | 3.0 s |
| `/api/health` probe p99 / max while loaded | 2.6 s / 7.4 s | 269 ms / 633 ms | **47 ms / 233 ms** |
| server RSS max | 403 MB | 169 MB | 152 MB |

Every stream had `: connected` padding, `token` events and a final `done` carrying `llm.label`.

### Scenario A (200 concurrent connections, 10 s each, 1 worker)

| endpoint | before (rps, p50 / p95) | after (rps, p50 / p95) |
|---|---|---|
| `GET /api/health` | 193, 541 ms / 4.5 s | 495, 267 ms / 1.1 s |
| `GET /api/config` | 356, 371 ms / 1.7 s | 449, 300 ms / 1.3 s |
| `GET /api/subjects` | 30, 7.3 s / 10.1 s (cold guests) | 251, 446 ms / 2.6 s (guests pre-warmed) |
| `GET /` static shell | 103, 905 ms / 5.5 s | 469, 273 ms / 1.3 s |

The "before" and "after" runs were not under identical machine load and the subjects row also changed
methodology (the first run included every guest's first-touch Mongo round trips), so read this table as
"cheap endpoints are not the problem", not as a precise speed-up. The client (Python on the same
laptop) is itself a limit at ~500 rps.

### Scenarios C, D, E (final code)

| | C: fault injection | D: 4 workers | E: 8 uploads |
|---|---|---|---|
| requests / 5xx / other errors | 2000 / 0 / 0 | 2000 / 0 / 0 | 32 / 0 / 0 |
| wall | 68.9 s | 74.9 s (B was 70.2 s) | 13.4 s |
| chat TTFT p50 / p95 | 3.2 s / 6.3 s | 2.5 s / 9.5 s | n/a |
| `/api/health` p99 | 251 ms | 331 ms | 82 ms |
| RSS | 140 MB | 535 MB (4 workers) | 189 MB |

* **C**: Anthropic returned 529 for 24 of 45 calls and was then skipped by the circuit breaker (3
  consecutive failures open it for 60 s); Groq returned 429 on 475 of 1643 calls. All 600 streams and
  400 JSON generations succeeded, answered by Groq key 1 (`gpt-oss-120b` mostly, `gpt-oss-20b` and
  `qwen3.8-27b` as the next models) or Groq key 2, a handful by Gemini. p95 latency was unchanged
  versus B: fall-through costs about one fast failed call, and the breaker removes even that. No 5xx
  leaked. Each answer's `Model:` label reflected the actual step.
* **D**: four workers gave **no speed-up** because the run is bound by the MongoDB cluster, not by
  Python (next section). Memory is ~110-135 MB per worker.
* **E**: each 1.5 MB upload costs about 1.3 s of CPU (pypdf, chunking, BM25, 768-d embeddings); 8
  concurrent uploads queue on the GIL (p50 10 s). Event-loop stayed healthy (health p99 82 ms) because the work runs in
  worker threads. On Vercel each instance handles its own upload.

### The ceiling: Atlas M0 throttles at ~110 ops/s

A pure `find_one` loop against the cluster: 5 threads 78 ops/s (p50 43 ms), 20 threads 109 ops/s
(p99 2.5 s), 50 threads 115 ops/s (p99 6.2 s). Before section 4a one realistic session issued **31 Mongo operations**
(counted with a pymongo command listener): subjects 3 (first call), history 1, chat start 1, each chat
message 2-4, quiz 1, flashcards 1, history 1, progress summary **12** (one `find` per artifact kind).
110 ops/s / 31 ops = ~3.5 sessions/s = ~57 s for 200 sessions, which is what B, C and D all measured
(~67-75 s, independent of worker count). The latency growth on trivial endpoints in B/D (p50 2-4 s) is
requests queueing for that shared budget. A paid Atlas tier (M10+) has no such cap; the application
itself was not the limit in these runs (health p99 stayed under 50 ms while 200 users were active).

## 3. Cold start and MongoDB

The first `/api/health/deep` of a fresh uvicorn process took 13.3 s in an earlier run. Findings
(phase-timed with a scratch script; CPU vs wall separated with `process_time`):

| phase (quiet-ish machine) | wall | CPU |
|---|---|---|
| `import pymongo` | ~100-280 ms | 100-190 ms |
| SRV + TXT lookup and `MongoClient()` construction | ~100-150 ms | 10-16 ms |
| first `ping` (A-records, TCP+TLS x3 members, auth, discovery) | ~480-540 ms | ~20 ms |
| second `ping` | 35-45 ms | |

So the floor is ~0.75 s, and nearly all of it is network round trips; CPU cost is tiny. Five fresh
processes **after** (load average 109-118): first `/api/health/deep` = 0.97, 1.11, 0.77, 0.81, 0.81 s;
second call 35-68 ms. The five **before** runs (load average 72, 72, 310, 310, 299) were 21.4, 14.2,
137.6 (a 503), 0.89 and 0.79 s: the outliers coincide with the machine being saturated (the same
`import dnspython` / `import pymongo` took 2-10 s in the instrumented runs), not with anything in the
code: the one-shot index-ensure thread was A/B tested (with and without) and made no difference; the
marker document makes it a single `find_one`. No code change was needed, none was made. Bounded
failure verified: with `MONGODB_URI` pointing at a blackhole address `/api/health/deep` reports
`ServerSelectionTimeoutError` and an authenticated route returns a clean 503 in 3.6 s (3 s selection
timeout + overhead), never a hang.

## 4. Bottlenecks found and fixed

1. **PDF text re-parsed on every request** (`backend/documents.py::read_pdf`): ~350 ms of GIL-bound pypdf CPU
   per quiz / flashcards / cheat-sheet / mind-map / exam / planner / teach-back call. 200 users x 2 calls
   starved the thread pools and everything (even simple history reads) hit 90 s timeouts. Now memoised
   per file version (path, mtime_ns, size), bounded (64 entries / 6 M chars), single-flight so a cold
   stampede parses once. This was the dominant fix (B: 314 s -> 69 s).
2. **Too few worker threads**: `asyncio.to_thread` uses the loop default executor (min(32, cpus+4): 5-6
   on a 1-2 vCPU instance) and sync routes use anyio's 40-token limiter. `backend/main.py` now raises
   both to `THREADPOOL_SIZE` (default 64) on the first request (and in lifespan).
3. **Mongo pool wait too short**: with `waitQueueTimeoutMS=5000`, bursts produced
   `WaitQueueTimeoutError` -> 503. Now 10 s (queue, don't fail). `maxPoolSize` stays 10 (configurable
   with `MONGO_MAX_POOL`; raising it to 20 did not help because the cluster is the limit).
4. **A history write could discard a paid generation**: if `db.save` failed (Mongo saturated) the
   request returned 503 after the LLM had already answered. `db.save` now logs and returns `None` for
   pymongo/bson errors (response goes out without a history id); other exceptions still raise.
5. `GET /api/subjects` now also returns `own` and `bundled` (the frontend uses them to show the delete
   button only on the caller's own subjects); `subjects` is unchanged.

6. **MongoDB operations per session cut from 31 to 12** (section 4a), which moves the M0 ceiling.

## 4a. Reducing MongoDB operations (the M0 budget is ~110 ops/s)

`scripts/loadtest.py` now counts every MongoDB command with a pymongo command-monitoring listener
(`scripts/_lt_server.py`, single worker). Per full session (subjects, history, chat with 3 messages, quiz,
flashcards, history, progress):

| operation | before | after | how |
|---|---|---|---|
| progress summary | 12 reads | **1** | one `$unionWith` aggregation, projected; 20 s per-owner cache |
| history lists (`GET /api/history/<kind>`) | 1 each time | 1, then cached | 15 s per-owner/kind/query cache |
| first `GET /api/subjects` | 3 (touch + 2 GridFS finds) | **2** | one aggregation over both GridFS file collections |
| chat start | 1 | 1 | unchanged (creates the chat doc) |
| each chat message | read + 1-3 writes | **1 write** (+1 title update on turn 1) | write-through chat-state cache (no read), title merged into the single `$push` update |
| quiz / flashcards save | 1 | 1 | unavoidable |
| **session total** | **31** | **12** | measured: 2401 ops / 200 sessions |

Scenario B with the same 200 users: wall 70 s -> **39 s**, chat TTFT p50 2.9 s -> **0.74 s** (p95 6.7 s ->
1.6 s), quiz p95 5.7 s -> 3.8 s, 0 errors. Theoretical ceiling at ~110 ops/s: 3.5 -> **~9 full sessions/s**.

**Scenario F, idle dashboard refresh** (200 users keep progress + two history lists + subjects open and
re-fetch every ~4 s for 6 rounds = 4800 requests): **8 ops per user for 24 requests** (mostly the first
fetch; steady state is about 0.18 ops/s per open dashboard, i.e. one aggregate per 20 s and one find per 15 s per
list). Before the changes the same refresh cost ~14 ops every 4 s (about 3.5 ops/s per open dashboard), so
M0 supported ~30 open dashboards; it now supports roughly 600. Request p50 was 2-9 ms (served from cache).

Correctness guards (tests in `tests/test_mongo_ops.py`, `tests/test_progress.py`): every cache is keyed by
owner so one owner never sees another's entry; each write for an owner bumps that owner's epoch *after* the
write, so the owner's next read is fresh on this instance; across instances a cache can be stale for at most its
TTL (15-20 s for lists/progress, 120 s for chat context; the stored chat transcript is never affected). Caches
are bounded (512-1000 entries) and return deep copies. `explain()` on every hot query shows an `IXSCAN`
and no collection scan; new indexes: GridFS `files` on (owner, subject) and (subject_key, owner), index
schema `idx-v4`.

Remaining floor for the heavy session: 6 writes that cannot be avoided (3 chat turns, quiz, flashcards) +
chat create + title update + guest validation + subjects + 2 history + progress. Further options (not done):
create the chat lazily on its first message (-1), fold the LLM title into the first write (-1).

## 5. Live end-to-end with the real providers (minimal spend)

A local server (`uvicorn api.index:app`) with real keys and a throwaway guest. One request each, all
HTTP 200 with parseable JSON and a `llm.label`:

| call | time | model |
|---|---|---|
| config / guest create / subjects / history / progress | 0.01-1.98 s (guest create is the first Mongo touch) | n/a |
| `chat/start` (Mathematics) | 0.3 s | n/a |
| `chat/message/stream` | TTFT 2.0 s, total 5.3 s, 44 tokens, 4 steps, `: connected` padding present | anthropic claude-sonnet-5-5 step 1 |
| quiz (3) / flashcards (3) / cheatsheet / mindmap | 7.7 / 5.5 / 11.3 / 4.3 s | anthropic step 1 |
| planner summaries / plan | 8.9 / 16.1 s | anthropic step 1 |
| exam (2 mcq + 1 short) / teach-back start / insights | 5.1 / 2.3 / 7.0 s | anthropic step 1 |
| progress record + summary, history list | 0.04-0.33 s | n/a |

**Fall-through, live and free:** with `ANTHROPIC_API_KEY` replaced by an invalid value in that process
only, the same chat stream was answered by `Model: groq · openai/gpt-oss-120b · step 2 (Groq key 1)`
(TTFT 1.6 s, total 3.3 s). The throwaway guests' data was deleted by guest id afterwards. The server log
showed no non-2xx responses for these flows.

SPA boot trace (headless browser against the real backend): a visitor with no session triggers
`GET /api/subjects` from `frontend/assets/app.js` (~line 1024, the home-page subject count) **before**
signing in or choosing guest, which is the logged **401**. It is harmless (the SPA ignores it) but noisy.
Suggested frontend fix (not applied; `frontend/` is not mine): only call it when `Auth.isGuest()` or the
user is signed in. I could not reproduce the **400**; the only 400 the backend can emit on the boot path is
`GET /api/guest/me` with an empty `X-Guest-Id`, which `auth.js` already avoids, so it was most likely a stale
stored guest id (the SPA clears the id on 400/404 and recovers).

## 6. Remaining limits (what real traffic will hit first)

* **Provider quotas, not the app, cap real LLM throughput.** The load test used fake providers, so real
  provider behaviour at scale is *not measured*. Known limits: Groq free tier 8,000 tokens per minute and
  200,000 per day per org (two keys help only if they are different orgs); Gemini free tiers are small
  and Gemini 503 "high demand" is common; Anthropic is paid and rate-limited per tier.
* **How the chain degrades.** Steps run Anthropic, Groq key 1 (3 models), Groq key 2, Gemini key 1, Gemini
  key 2. A per-minute 429 falls through immediately (it waits only if `retry-after` is at most 2 s); a
  daily-quota 429 is remembered for 10 minutes so later requests skip that (step, model) at once; three
  consecutive failures open a 60 s circuit breaker; a 401/403 marks the key dead for 10 minutes. A user
  sees a slightly slower or lower-tier answer with a different `Model:` label. When every step is
  exhausted they get HTTP 503 with a friendly message and `Model: none (all providers failed)`; streams
  that already started a provider never switch mid-answer. Prompts marked sensitive never go to Gemini.
  Each chat message costs ~1 streamed call plus a title call on the first turn, a quiz/flashcards ~1;
  with Groq's 8,000 TPM a handful of simultaneous document-sized prompts will trip it, which is the
  intended trigger for fall-through.
* The MongoDB M0 throttle above (~110 ops/s) now caps database-backed endpoints at roughly 9 full sessions
  per second (was 3-4) and ~600 open idle dashboards.
* A single instance saturates its CPU on uploads (1.3 s per 1.5 MB PDF) and on 200 simultaneous quiz
  generations; Vercel scales out instances instead.
* `.env` still holds legacy `LLM_PROVIDER`, `DEFAULT_MODEL` and `AVAILABLE_MODELS` (old Groq model names)
  which only appear in the local startup banner; remove them to avoid confusion.

## 7. Recommendations

* Use **Vercel Fluid Compute** (in-function concurrency): one warm instance serves many mostly-waiting
  requests, which matches this I/O-bound workload and shares the PDF-text and subject caches.
* Add **Vercel Firewall rate limits**: `POST /api/guest` (the per-instance throttle in `routers/guest.py`
  is only a first line), `POST /api/subjects` (CPU-heavy upload), and the `/api/*/generate` routes per IP.
* Move MongoDB off M0 before real traffic (or accept ~110 ops/s; see section 4a).
* Put the app on paid provider tiers (or add keys from different organisations) for production use;
  keep Anthropic as step 1 and watch the `Model:` labels / `skipped` lists in logs to see fall-through rates.

## 8. Round 2: the spec run (200 users, faults, chunked uploads, retrieval)

Run on the same laptop and the real Atlas M0 cluster, with fake providers. The machine was shared
with Chrome (GPU and renderer helpers at 100-180 % CPU) and VS Code (plugin host at 80-115 % CPU),
and `uptime` load averages were 4-11 on 8 cores. The "before" runs were the more contended ones (client
event-loop lag p99 0.8-2.5 s against 0.2-0.3 s later), so treat the wall-clock and TTFT deltas as indicative.
The attributable deltas are the Mongo operation counts, the micro-benchmarks and the server-side timings.

Commands used (ports are fixed: API 8911, fake providers 8912; the run refuses to start if either is taken):

```bash
./venv/bin/python scripts/loadtest.py --scenarios b --users 200                    # realistic sessions
./venv/bin/python scripts/loadtest.py --scenarios c --users 200                    # faults: Anthropic 50% 529, Groq 30% 429, Gemini 100% 429
./venv/bin/python scripts/loadtest.py --scenarios g --g-users 20 --g-gemini-fail 1.0 --g-recover-s 20   # chunked uploads, Gemini outage that ends
./venv/bin/python scripts/bench_retrieval.py concurrent | cold | quota            # retrieval, 200 queries across both bundled subjects
LT_SAMPLE=1 ./venv/bin/python scripts/loadtest.py ...                             # adds an in-process thread sampler (event-loop profile)
```

### 8.1 Results table

| scenario | load | result | errors | p50 / p95 / p99 |
|---|---|---|---|---|
| **A** realistic session (`loadtest.py -s b`) before fixes, run 1 | 200 users, 2000 req, 126 s | TTFT | 0 5xx, 1 read error | TTFT 10.7 / 40.8 s; `/api/health` p99 4.7 s |
| **A** before fixes, run 2 | 200 users, 98 s | TTFT, server-side TTFB | 0 5xx, 2 connect errors (keep-alive race, harness fixed) | TTFT 7.8 / 23.5 s; TTFB p95 1.59 s; health p99 2.3 s |
| **A** after fixes, runs 1 and 2 | 200 users, 2000 req, 46 s each | TTFT, server TTFB | **0 5xx, 0 errors** | TTFT 1.8 / 6.9 s and 1.7 / 6.7 s; total p95 8.1 s and 7.7 s; TTFB p95 0.55 s; health p99 0.56 and 0.49 s; client loop lag p99 0.25 s |
| server-side per endpoint, after | 200 users | `GET /api/subjects` / `POST chat/start` / quiz generate | - | server p50 2.0 / 1.3 / 0.8 s, p95 4.3 / 2.9 / 1.9 s (Mongo queueing) |
| Mongo ops per session | - | before 12.8 (`update:guests` 370 per 200), after 12.0 (200) | - | command RTT p50 46-54 ms, p95 180-320 ms, p99 350-690 ms |
| **B** faults, before the CMap/sniffio/touch fixes | 200 users, 1993 req, 124 s | Anthropic 529 and Groq 429 injected; Gemini 100 % 429 | 1 HTTP 503 (designed), 5 read errors (harness keep-alive race, now fixed) | TTFT 8.6 / 25.2 s; total p95 27.1 s |
| **B** faults, after all fixes | 200 users, 2000 req, 46 s | injected: Anthropic 493 of 939 calls (52 %), Groq 316 of 1069 (30 %), Gemini 78 of 78 | **1 HTTP 503 (designed "all providers failed") of 2000, 0 other 5xx, 0 client errors** | TTFT 1.5 / 6.1 s; total p95 7.6 s; breaker opened 82x, closed 79x |
| **C** chunked uploads (`loadtest.py -s g`), 20 users, Gemini 429 for 20 s then healthy (before CMap memo) | 20 uploads, 60 + continue calls | partial at completion 20/20 (0 embedded), complete 20/20 after recovery | 0 5xx | upload end-to-end 9.6 / 10.5 s; `complete` server 7.8 s; continue-index 25 calls p50 per subject |
| **C** after the CMap memo | same | same partial then complete 20/20 | 0 5xx | upload end-to-end 7.5 / 8.4 s; `complete` server 5.1 s; continue-index p50 373 ms |
| **D** 200 concurrent queries, mixed repeats, 2 subjects | 40 threads | hybrid 180, dense 20 | - | completion 270 / 311 / 392 ms; service 26 ms p50 |
| **D** 200 concurrent queries, all unique | same | hybrid 184, dense 16 | - | completion 439 / 710 / 748 ms; service 136 ms p50; 266 q/s |
| **D** cold instance, first query (fresh interpreter, median of 3) | 1 query | index load 75 ms, import 48 ms | - | first query 222 ms, second 126 ms; RSS 28 -> 50 MB |
| **D** both Gemini keys 429 | 60 concurrent + 20 sequential | 60/60 answered lexically (BM25), 42 upstream calls; while the breaker is open 20 sequential queries cost 0.1 ms and 0 calls | 0 errors to the user | outage p50 115 / p99 221 ms; recovered 21.3 s after the provider healed (breaker closed) |
| Mongo M0 ceiling (probe, same cluster) | `find_one` | 1 thread: 24 ops/s (p50 35 ms); 10 threads: 99 ops/s (p95 389 ms); 30 threads: 97 ops/s (p95 2.0 s) | - | ceiling ~100 ops/s confirmed; 2400 ops in 46 s = 52 ops/s average after the fixes (26 before) |
| server RSS | | B: 150-190 MB max (1 worker); D retrieval: 53 MB | - | - |

Not run as separate scenarios: static shell / health at 200 connections (`-s a`): the health probe
inside scenario B (p99 0.5 s, under full load) covers it; `e` (8 multipart uploads) is superseded by `g`;
`d` (4 workers) was not repeated (round 1 showed no speed-up because the cluster is the limit).

### 8.2 Bottlenecks found and fixed in round 2

1. **PDF CMap re-parsing on every page** (`backend/documents.py`, `_install_cmap_memo`): pypdf parses a
   font's ToUnicode CMap again for each page that uses it. About half of the CPU of indexing three
   4-page lectures went to `parse_bfrange`. A bounded (256 entries), process-wide memo keyed by
   SHA-256(CMap bytes + subtype) returns copies. Extraction CPU for 12 four-page PDFs **1215 -> 673 ms**
   (-45 %), output byte-identical to stock pypdf (tests: `tests/test_pdf_cmap_memo.py`). Upload
   `complete` server time p50 **7.8 -> 5.1 s** under 20 concurrent uploads.
2. **httpcore re-probes the event loop with a failing `import trio` on every async lock**: without
   `sniffio` installed, `current_async_library()` walks `sys.path` on the event loop (**206 us per call**,
   about 5 % of the sampled loop thread). `sniffio` is now a declared runtime dependency (`requirements.txt`):
   **0.9 us per call**.
3. **Guest `lastActiveAt` written once a minute**: a 2-minute session wrote the guest twice. The touch cache
   is now 5 minutes (the TTL is 30 days). `update:guests` **370 -> 200** per 200 sessions, ops per session
   12.8 -> 12.0 (tests: `tests/test_loadtest_fixes.py`).
4. **Stale index cache hid the embedding path** (test harness, found while testing): the loadtest cleanup
   did not remove `vector_store._cache_root()/user__guest_loadtest*`, so the next upload with the same chunk
   text reused old vectors and made **zero** Gemini calls. Cleanup now removes these (`index_cache_dirs`).
5. **Breaker observability**: `llm_chain._state` now counts opens and closes (`breaker_opened`,
   `breaker_closed`); the load test reports them (C: 82 opens, 79 closes).
6. **Harness fixes**: keep-alive expiry 3 s on the client (uvicorn closes idle sockets at 5 s; a reused
   closed socket showed as `ReadError`); exception texts are reported; a port guard; the bench recovery
   check accepted only `hybrid`, which is a false negative when the probe text has no lexical hit (now any
   embedding-backed mode counts).

Tried and **not** changed: `MONGO_MAX_POOL` 30 vs 10 (two runs each: 67 / 60 s vs 98 / 61 s, server TTFB p95
518 vs 806 ms): inconclusive on this machine, and a higher pool multiplies connections across serverless
instances against the M0 connection limit, so the default stays 10. The event loop itself was not saturated
in any run (the sampled leaves were uvloop's C select, and Mongo socket waits in threads).

### 8.3 Not measurable locally

* **Real provider rate limits**: Groq free tier (8,000 TPM, 200,000 TPD per org), Gemini free-tier RPM and
  daily caps, Anthropic tier limits and real 529 bursts. The fake returns what we configure; daily-quota
  classification (`_is_daily`) is only exercised by unit tests.
* **Network**: laptop to Atlas, and provider TLS, round trips, not a datacentre path. Cluster latency is real.
* **Scale-out**: Vercel instance count, cold starts and Fluid concurrency. This was one process (plus the
  4-worker run of round 1).
* **Quality**: fake embeddings are random vectors, so retrieval numbers are speed and behaviour only.

## 9. Round 3: database/state layer (200 users, real Atlas M0, fake LLMs, ports via `LT_API_PORT` / `LT_FAKE_PORT`)

| | before | after |
|---|---|---|
| Mongo ops per full session (scenario b) | 12.0 (`update:guests` 200, `aggregate:uploaded_pdfs.files` 200, `find:chats` 200, `insert:chats` 200, `update:chats` 800, `insert` 400, `find:quizzes` 200, `aggregate:attempts` 200) | **7.0** (`findAndModify:guests` 200, `aggregate:chats` 200 (owner snapshot), `update:chats` 600, `insert` 400) |
| wall for 200 sessions / sessions per second | 39.7 s / 5.0 | 32.9 s / 6.1 (machine load avg ~10) |
| server-side p50 history / progress / subjects | 609-1994 ms / 575 ms | 1 / 1 / 129 ms (first touch only) |
| scenario f ops per user (24 requests) | 7.9 | **2.0** (the guest touch + one snapshot read) |
| scenario f request p95 | 333-922 ms | 1-128 ms |

Floor now: 1 guest validation, 1 snapshot read, 3 chat-turn writes, quiz and flashcards saves = 7. With a shared cache
(DEPLOYMENT.md) the guest validation is also skipped on instance hops. `tests/test_cache.py::TestSessionOpBudget` fails if a
session exceeds 7 ops. Absolute latencies are noisy (shared laptop); the op counts are exact.


## 10. Ops cost of the Anthropic spend guards (`backend/spend.py`)

Not part of the 7-op session floor above; it is amortised per instance and per active owner, not per request:

| Operation | Rate | Cost |
|---|---|---|
| read shared totals (`find` on `usage_daily`, `_id $in` [global, owner], projection `{tokens: 1}`) | at most once per owner per 30 s, only when that owner is about to make a paid-model call | 1 op |
| flush (`bulk_write` of `$inc` upserts, all scopes in one round trip) | per instance every ~20 s while it has pending usage, or at 5% of a budget, plus shutdown | 1 op |
| budgets set to 0, or no `ANTHROPIC_API_KEY` | never | 0 ops |
| shared Redis enabled | one pipeline (`INCRBY`+`EXPIRE`) per flush, one `MGET` per read | 0 Mongo ops |

A 10-minute chatty session therefore adds about 20 reads (only while it keeps calling the paid model) and its instance
shares about 30 flushes with everyone else on that instance. `tests/test_spend.py::test_mongo_ops_per_session_is_one_projected_read_per_30s_and_one_bulk_write_per_flush`
pins this. With the fake provider (`LLM_CHAIN_TEST_BASE_URL`) the budgets apply as usual; set
`ANTHROPIC_DAILY_TOKEN_BUDGET=0` for load runs that must not fall through to the free steps.
