"""Pre-download the two Hugging Face models so the demo runs offline afterwards.

Run once with internet: `python scripts/download_models.py`. Models land in the normal
Hugging Face cache (~/.cache/huggingface), about 220 MB in total.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"  # this script is the one place allowed to download

from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if __name__ == "__main__":
    SentenceTransformer(EMBED_MODEL, device="cpu")
    print("ok:", EMBED_MODEL)
    CrossEncoder(RERANK_MODEL, device="cpu")
    print("ok:", RERANK_MODEL)
