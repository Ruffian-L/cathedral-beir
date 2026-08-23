# Cathedral BEIR

**Pure dense retrieval** experiments on BEIR-style workloads: single embedding + cosine, no BM25 or rerank as the default path.

**Lead:** Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L))

Twin to [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench): same person, two retrieval stacks, pick by evidence.

## Method

| | |
|--|--|
| **Simple method** | Nomic Embed-class dense only · L2 normalize · cosine / matmul |
| **Repro** | `python benchmark.py` (+ per-dataset flags) |

1. Embed corpus with `nomic-ai/nomic-embed-text-v1.5` (768-d, normalized)
2. Embed queries with the model’s query prefix (e.g. `search_query:`)
3. Retrieve by cosine (dot product of L2-normalized vectors)
4. Score with BEIR-style nDCG@k / Recall@k

No BM25. No cross-encoder rerank. No learned sparse.

## Project-run numbers

Measurements from this harness. Re-run to refresh.

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

Average over these rows: ~0.54.

## Reproduction

Python 3.10+ · CUDA GPU recommended (8GB+; 16GB for large corpora) · disk for datasets

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

## Findings so far

- Embedding quality + correct query prefix can carry a pure dense stack far.
- Normalization makes retrieval a matmul.
- Hybrid is a choice; see SplatRagBench.

## Citation

Cite the repo + commit.

```bibtex
@misc{cathedral_beir,
  title={Cathedral BEIR — pure dense retrieval experiments},
  author={Van Pham, Jason},
  year={2025},
  howpublished={https://github.com/Ruffian-L/cathedral-beir}
}
```

## License

MIT

## Acknowledgments

- [BEIR](https://github.com/beir-cellar/beir) · [Nomic AI](https://www.nomic.ai/) · Sentence Transformers
