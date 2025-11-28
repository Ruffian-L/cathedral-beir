#!/usr/bin/env python3
"""
Cathedral BEIR Benchmark

Pure 768D cosine similarity. No reranking. No sparse. No bullshit.

Results:
  SciFact:    0.7036 nDCG@10
  NFCorpus:   0.3381 nDCG@10
  TREC-COVID: 0.7226 nDCG@10
  AVERAGE:    0.5881 nDCG@10

SOTA 2025: ~0.52 avg nDCG@10 (hybrid dense+sparse+reranking)
"""

import argparse
import json
import logging
import os
import time
import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = ["scifact", "nfcorpus", "trec-covid"]

def download_dataset(dataset_name: str, data_dir: str = "datasets"):
    """Download BEIR dataset if not exists."""
    data_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_path):
        logger.info(f"Downloading {dataset_name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, data_dir)
    return data_path

def load_dataset(dataset_name: str, data_dir: str = "datasets"):
    """Load BEIR dataset."""
    data_path = download_dataset(dataset_name, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    logger.info(f"Loaded {dataset_name}: {len(corpus)} docs, {len(queries)} queries")
    return corpus, queries, qrels

def embed_texts(model, texts: list, batch_size: int = 8, prefix: str = "") -> np.ndarray:
    """Embed texts with optional prefix."""
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

def run_benchmark(datasets: list, output_file: str = "results.json"):
    """Run Cathedral BEIR benchmark."""
    
    logger.info("=" * 70)
    logger.info("     CATHEDRAL BEIR BENCHMARK")
    logger.info("     Pure 768D cosine. No reranking. No bullshit.")
    logger.info("=" * 70)
    
    # Load model once
    logger.info("\nLoading Nomic v1.5 (768D)...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
    all_results = {}
    
    for dataset in datasets:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"     DATASET: {dataset.upper()}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load dataset
            corpus, queries, qrels = load_dataset(dataset)
            corpus_ids = list(corpus.keys())
            query_ids = list(queries.keys())
            
            # Prepare texts
            corpus_texts = [
                f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
                for doc_id in corpus_ids
            ]
            query_texts = [queries[qid] for qid in query_ids]
            
            # Adaptive batch size for large datasets
            batch_size = 8 if len(corpus) > 50000 else 32
            
            # Embed corpus (in chunks for large datasets)
            logger.info(f"Embedding {len(corpus)} documents...")
            if len(corpus) > 50000:
                # Chunked embedding for large datasets
                chunk_size = 10000
                all_embeddings = []
                for i in range(0, len(corpus_texts), chunk_size):
                    chunk = corpus_texts[i:i+chunk_size]
                    logger.info(f"  Chunk {i//chunk_size + 1}/{(len(corpus_texts)-1)//chunk_size + 1}")
                    emb = embed_texts(model, chunk, batch_size=batch_size)
                    all_embeddings.append(emb)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                corpus_emb = np.vstack(all_embeddings)
            else:
                corpus_emb = embed_texts(model, corpus_texts, batch_size=batch_size)
            
            # Embed queries (with search_query prefix for Nomic)
            logger.info(f"Embedding {len(queries)} queries...")
            query_emb = embed_texts(model, query_texts, batch_size=32, prefix="search_query: ")
            
            # Retrieve with pure cosine similarity
            logger.info("Retrieving with pure 768D cosine similarity...")
            results = {}
            similarities = query_emb @ corpus_emb.T
            
            for i, qid in enumerate(query_ids):
                scores = similarities[i]
                top_indices = np.argsort(scores)[::-1][:100]
                results[qid] = {
                    corpus_ids[idx]: float(scores[idx])
                    for idx in top_indices
                }
            
            # Evaluate
            evaluator = EvaluateRetrieval()
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
            
            elapsed = time.time() - start_time
            
            all_results[dataset] = {
                "nDCG@10": ndcg["NDCG@10"],
                "Recall@10": recall["Recall@10"],
                "time": elapsed,
                "docs": len(corpus),
                "queries": len(queries)
            }
            
            logger.info(f"\n📊 Results for {dataset}:")
            logger.info(f"   nDCG@10:  {ndcg['NDCG@10']:.4f}")
            logger.info(f"   Recall@10: {recall['Recall@10']:.4f}")
            logger.info(f"   Time:     {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Error on {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}
    
    # Calculate average
    valid_results = [r for r in all_results.values() if "nDCG@10" in r]
    if valid_results:
        avg_ndcg = np.mean([r["nDCG@10"] for r in valid_results])
        avg_recall = np.mean([r["Recall@10"] for r in valid_results])
        all_results["average"] = {"nDCG@10": avg_ndcg, "Recall@10": avg_recall}
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n💾 Results saved to {output_file}")
    
    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info("                     FINAL SUMMARY")
    logger.info("=" * 70)
    
    for dataset, metrics in all_results.items():
        if dataset == "average":
            continue
        if "error" in metrics:
            logger.info(f"{dataset:15} ERROR: {metrics['error']}")
        else:
            logger.info(f"{dataset:15} nDCG@10: {metrics['nDCG@10']:.4f}")
    
    if "average" in all_results:
        logger.info("-" * 70)
        logger.info(f"{'AVERAGE':15} nDCG@10: {all_results['average']['nDCG@10']:.4f}")
    
    logger.info("=" * 70)
    logger.info("     SOTA 2025: ~0.52 avg nDCG@10 (hybrid dense+sparse+reranking)")
    logger.info("     Cathedral: Pure 768D cosine. No reranking. No bullshit.")
    logger.info("=" * 70)
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cathedral BEIR Benchmark")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, 
                       choices=DATASETS + ["all"],
                       help="Datasets to benchmark")
    parser.add_argument("--output", default="results.json", help="Output file")
    args = parser.parse_args()
    
    if "all" in args.datasets:
        args.datasets = DATASETS
    
    results = run_benchmark(args.datasets, args.output)
