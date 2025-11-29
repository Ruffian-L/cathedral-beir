# HotpotQA (5.23M docs) — Cathedral Engine Run

This file records the run details for `HotpotQA` (distractor) executed with the Cathedral Engine v2 (Gravity Edition / Matryoshka / FAISS) and saved to the `cathedral_beir/` repo.

## Run Summary
- Date: 2025-11-29
- Dataset: HotpotQA (distractor)
- Corpus size: 5,233,329 documents
- Queries: 7,405

## Hardware & Environment
- GPU: RTX 5080 Mobile (16GB), CUDA available; FAISS GPU attempted but fell back to CPU (faiss.Python issue 'StandardGpuResources' not available).
- Environment flags: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Model: `nomic-ai/nomic-embed-text-v1.5` (Matryoshka truncation used in some runs)

## Settings used
- Matryoshka dim: 512D or 768D as per your run command (ambiguous) - use `--matryoshka_dim` in run command to be explicit
- Batch size: 1536 (fallbacks exist during wrapper runs) in previous wrapper runs; this run used 1536→512 fallback logic in batch wrapper script
- FP16: True
- FAISS GPU: attempted; CPU fallback used
- Hybrid BM25: off (pure dense retrieval) for these initial runs
- Gravity: default (0/none) for the pure dense baseline; use `--gravity` to enable black-hole gravity

## Timings
- Corpus embedding time: 103.7 minutes (reported)
- Throughput: 841 docs/sec (reported)
- Index build time: 23.1s
- Search time: 751.5s (for all queries/search)

## Results
- nDCG@1: 0.7805
- nDCG@3: 0.6703
- nDCG@5: 0.6951
- nDCG@10: 0.7151
- nDCG@100: 0.7453
- Recall@10: 0.7493
- Recall@100: 0.8674

## Notes & Observations
- FAISS GPU module call `StandardGpuResources()` is unavailable in this environment, so CPU offline search was used for indexing and search.
- Query encoding was very fast (few seconds), indicating possible use of small batches or cached tokenizer/encoder. Confirm batch sizes for queries if exact timings are critical.
- Indexing 5.23M vectors using FAISS CPU took ~23s for `IndexFlatIP` add() (memory backed). Searching top-100 across all queries took ~12.5 minutes (751.5s).

## Next Steps
- Add the HotpotQA results to the main leaderboard (done for `cathedral_beir/README.md`).
- Optionally re-run with Gravity (1.8, 1.6) and re-test hybrid BM25 fusion to measure gains and produce a small comparative table.
- Consider FAISS GPU fix or CPU/GPU environment parity for faster indexing/search on large corpora like HotpotQA.

---

Generated automatically during the Cathedral Engine v2 runs.
