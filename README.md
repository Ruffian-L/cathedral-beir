# Cathedral BEIR

**Pure dense retrieval** experiments on BEIR-style workloads — single embedding + cosine, no BM25 / rerank theater as the default path.

**Lead:** Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L))  
**Built with:** Grok · Claude · Gemini (and related tools where they helped)

> **Not** a paper claiming official BEIR leaderboard crown.  
> **Not** hybrid SOTA deathmatch copy.  
> **Yes** a clean “simple route when the scoreboard says so” twin to [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench).

## Best face of this repo

| Strength | Where |
|----------|--------|
| **Simple method** | Nomic Embed-class dense only · L2 normalize · cosine / matmul |
| **Repro scripts** | `python benchmark.py` (+ per-dataset flags) |
| **Duality thesis** | Same person ships hybrid physics-RAG **and** pure dense — pick by evidence |
| **What it is not** | Independent MTEB/BEIR certification; numbers are **project runs** |
| **Honesty** | Prefer in-repo tables + re-run over marketing “beats SOTA” headlines |

## Method (the whole recipe)

1. Embed corpus with `nomic-ai/nomic-embed-text-v1.5` (768-d, normalized)  
2. Embed queries with the model’s query prefix (e.g. `search_query:`)  
3. Retrieve by cosine (dot product of L2-normalized vectors)  
4. Score with BEIR-style nDCG@k / Recall@k  

No BM25. No cross-encoder rerank. No learned sparse. No hyperparameter sweep theater.

## Project-run numbers (as recorded in this tree)

Treat as **this harness’s measurements**, not frozen external leaderboard truth. Re-run to refresh.

| Dataset | Corpus (approx.) | Pure dense nDCG@10 (this engine) |
|--------|------------------:|---------------------------------:|
| Quora | 522K | 0.8818 |
| TREC-COVID | 171K | 0.7226 |
| HotpotQA (distractor) | 5.23M | 0.7151 |
| SciFact | 5K | 0.7036 |
| ArguAna | 8.6K | 0.3934 |
| FiQA | 57K | 0.3745 |
| NFCorpus | 3.6K | 0.3381 |
| SciDocs | 25K | 0.1865 |

**Average over listed rows in this README’s historical table:** ~0.54 (see full notes in git history / artifacts if present).  
Any “vs SOTA / hybrid” columns in older drafts were **illustrative** and are **not** a publishable comparative claim without matched protocol and third-party replication.

## Reproduction

**Needs:** Python 3.10+ · CUDA GPU recommended (8GB+; 16GB for large corpora) · disk for datasets

```bash
git clone https://github.com/Ruffian-L/cathedral-beir.git
cd cathedral-beir
pip install -r requirements.txt

python benchmark.py
python benchmark.py --datasets scifact
python benchmark.py --datasets nfcorpus
python benchmark.py --datasets trec-covid
```

Rough runtime (hardware-dependent): SciFact / NFCorpus ~minutes; TREC-COVID much longer.

## Key findings (scoped)

- **Embedding quality + correct query prefix** can carry a pure dense stack far.  
- **Normalization** makes retrieval a matmul.  
- **Complexity is not free** — hybrid is a choice, not a religion (see SplatRagBench).

## Citation

If you cite this tree, cite the **repo + commit** and call results **project evaluations**.

```bibtex
@misc{cathedral_beir,
  title={Cathedral BEIR — pure dense retrieval experiments},
  author={Van Pham, Jason},
  year={2025},
  howpublished={\url{https://github.com/Ruffian-L/cathedral-beir}},
  note={Project evaluation; not an official BEIR leaderboard entry}
}
```

## License

MIT

## Acknowledgments

- [BEIR](https://github.com/beir-cellar/beir) · [Nomic AI](https://www.nomic.ai/) · Sentence Transformers  
