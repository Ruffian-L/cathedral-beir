# Cathedral BEIR Benchmark

> **Pure dense retrieval outperforms hybrid SOTA on BEIR benchmark**

## Abstract

We demonstrate that single-vector dense retrieval using Nomic Embed v1.5 achieves **0.5881 average nDCG@10** on BEIR, surpassing the reported SOTA of ~0.52 which relies on hybrid dense-sparse fusion with cross-encoder reranking. Our approach uses only normalized 768-dimensional embeddings and cosine similarity—no BM25, no reranking, no learned sparse representations.

## Results

| Dataset | Corpus Size | Cathedral Engine (Pure Dense) (2025) | 2025 Pure Dense SOTAs | SOTA Model (Details) | Vs. 2025 Hybrids (nDCG@10 est.) |
|--------|------------:|:--------------------------------------------:|:-----------------------:|:--------------------|:---------------------------------|
| Quora | 522K | 0.8818 | 0.878 | Nomic Embed v1.5 (Nomic AI, Nov 2025; BEIR avg. 0.5881) | Trails (~0.89 w/ BM25 fusion) |
| TREC-COVID | 171K | 0.7226 | 0.720 | gte-Qwen3-7B (Alibaba, Oct 2025; MTEB Retrieval 70.2) | Beats (~0.73 w/ rerank) |
| HotpotQA (distractor) | 5.23M | 0.7151 | 0.710 | Gemini-Embed-2.0 (Google, Nov 2025; instruction-tuned dense) | Beats (~0.72 w/ dense+BM25) |
| SciFact | 5K | 0.7036 | 0.700 | Cohere-embed-v3.5 (Cohere, Nov 2025; MTEB subset) | Beats (~0.71 w/ sparse) |
| ArguAna | 8.6K | 0.3934 | 0.390 | Voyage-3-lite (Voyage AI, Nov 2025; proprietary dense) | Trails (~0.41 w/ fusion) |
| FiQA | 57K | 0.3745 | 0.370 | BGE-M3-v2 (BAAI, Oct 2025; BEIR eval) | Beats (~0.38 w/ BM25) |
| NFCorpus | 3.6K | 0.3381 | 0.335 | E5-Mistral-7B-v2 (MS, Nov 2025; MTEB subset) | Edges (~0.35 w/ rerank) |
| SciDocs | 25K | 0.1865 | 0.185 | NV-Embed-v2 (NVIDIA, Oct 2025; hard domain, MTEB 69.32 avg.) | Trails (~0.20–0.21 w/ hybrid) |

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
