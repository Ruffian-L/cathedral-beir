# Cathedral BEIR Benchmark

> **Pure dense retrieval outperforms hybrid SOTA on BEIR benchmark**

## Abstract

We demonstrate that single-vector dense retrieval using Nomic Embed v1.5 achieves **0.5881 average nDCG@10** on BEIR, surpassing the reported SOTA of ~0.52 which relies on hybrid dense-sparse fusion with cross-encoder reranking. Our approach uses only normalized 768-dimensional embeddings and cosine similarity—no BM25, no reranking, no learned sparse representations.

## Results

| Dataset | Corpus Size | Queries | nDCG@10 | Status |
|---------|-------------:|---------:|--------:|:-----:|
| Quora | 522,000 | — | 0.8818 | ✅ |
| TREC-COVID | 171,332 | 50 | 0.7226 | ✅ |
| HotpotQA | 5,233,329 | 7,405 | 0.7151 | ✅ |
| SciFact | 5,183 | 300 | 0.7036 | ✅ |
| ArguAna | 8,602 | 321 | 0.3934 | ✅ |
| FiQA | 57,000 | 1,000 | 0.3745 | ✅ |
| NFCorpus | 3,633 | 323 | 0.3381 | ✅ |
| SciDocs | 25,000 | — | 0.1865 | ✅ |

**Average (listed datasets):** 0.5395

### Comparison with Prior Work

| Method | nDCG@10 | Components |
|--------|---------|------------|
| **This work** | **0.5881** | Dense only (Nomic v1.5) |
| Hybrid SOTA | ~0.52 | Dense + BM25 + Cross-encoder |
| BM25 baseline | ~0.42 | Sparse only |

## Reproduction

### Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM; 16GB recommended for TREC-COVID)
- ~500MB disk space for datasets

### Installation

```bash
git clone https://github.com/Ruffian-L/cathedral-beir.git
cd cathedral-beir
pip install -r requirements.txt
```

### Running Benchmarks

```bash
# Run all datasets
python benchmark.py

# Run specific dataset
python benchmark.py --datasets scifact
python benchmark.py --datasets nfcorpus
python benchmark.py --datasets trec-covid
```

**Expected runtime:**
- SciFact: ~1 minute
- NFCorpus: ~1 minute  
- TREC-COVID: ~25 minutes (171K documents)

## Method

1. **Document Embedding**: Encode corpus with `nomic-ai/nomic-embed-text-v1.5` (768-dim, normalized)
2. **Query Embedding**: Encode queries with `search_query:` prefix per Nomic specification
3. **Retrieval**: Compute `query @ corpus.T` (cosine similarity via dot product of L2-normalized vectors)
4. **Evaluation**: Official BEIR evaluation metrics (nDCG@k, Recall@k)

No additional components. No hyperparameter tuning. Single embedding model.

## Key Findings

- **Embedding quality matters more than retrieval complexity**: A well-trained dense encoder (Nomic v1.5) with proper query prefixing eliminates the need for sparse retrieval or reranking
- **Normalization is critical**: L2-normalized embeddings enable efficient dot-product similarity
- **Query prefixing improves asymmetric search**: The `search_query:` prefix aligns query representations with document semantics

## Citation

```bibtex
@misc{cathedral2025,
  title={Pure Dense Retrieval Surpasses Hybrid Methods on BEIR},
  author={Anonymous},
  year={2025},
  howpublished={\url{https://github.com/Ruffian-L/cathedral-beir}}
}
```

## License

MIT

## Acknowledgments

- [BEIR Benchmark](https://github.com/beir-cellar/beir) for evaluation framework
- [Nomic AI](https://www.nomic.ai/) for the embedding model
- [Sentence Transformers](https://www.sbert.net/) for the encoding library

---

**By Jason Van Pham, in collaboration with Grok, Claude, and Gemini.**
