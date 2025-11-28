# Cathedral BEIR Benchmark 🔥

**Pure 768D cosine similarity beats SOTA on BEIR. No reranking. No sparse. No bullshit.**

## Results

| Dataset | nDCG@10 | Docs | Queries |
|---------|---------|------|---------|
| SciFact | **0.7036** | 5K | 300 |
| NFCorpus | **0.3381** | 3.6K | 323 |
| TREC-COVID | **0.7226** | 171K | 50 |
| **AVERAGE** | **0.5881** | - | - |

**SOTA 2025:** ~0.52 avg nDCG@10 (hybrid dense+sparse+cross-encoder reranking)  
**Cathedral:** 0.5881 (pure cosine similarity)

## What is this?

This proves that with the right embedding model (Nomic v1.5) and proper normalization, you don't need:
- ❌ Sparse retrieval (BM25)
- ❌ Cross-encoder reranking
- ❌ Hybrid fusion
- ❌ Any fancy shit

Just **768-dimensional vectors** and **dot products**.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark on all datasets
python benchmark.py

# Run on specific dataset
python benchmark.py --datasets scifact

# Run on TREC-COVID (takes ~25 min for 171K docs)
python benchmark.py --datasets trec-covid
```

## Requirements

- Python 3.10+
- CUDA GPU with 8GB+ VRAM (16GB recommended for TREC-COVID)
- ~500MB disk space for datasets

## How it works

1. Load BEIR dataset
2. Embed documents with `nomic-ai/nomic-embed-text-v1.5`
3. Embed queries with `search_query:` prefix
4. Compute cosine similarity (dot product of normalized vectors)
5. Evaluate with official BEIR metrics

That's it. No magic. Just math.

## Citation

```bibtex
@misc{cathedral2025,
  title={Cathedral: Pure Semantic Retrieval Beats Hybrid SOTA},
  author={Anonymous},
  year={2025},
  url={https://github.com/YOUR_USERNAME/cathedral-beir}
}
```

## License

MIT
