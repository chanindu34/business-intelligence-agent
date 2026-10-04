"""
Hybrid retrieval over the JKH Annual Report 2025/26 index.

    BM25 (stemmed keywords) + dense (Gemini embeddings)
      -> Reciprocal Rank Fusion (k = 60)
      -> optional cross-encoder rerank
      -> passages with their PDF page numbers

The index (chroma_db/) holds 820 chunks from PDF pages 1 to 60 and 139 to 160.
data/chunk_pages.json maps every chunk to the page it came from, so answers
can cite pages without re-embedding anything.
"""

import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "day10_documents"
PAGES_PATH = os.path.join(HERE, "data", "chunk_pages.json")
RRF_K = 60
NUM_CANDIDATES = 20
TOP_K = 5

# Small English cross-encoder (~90 MB). The previous version named
# "cross-encoder/mmarco-MiniLMv2-L12-H384-v1", which does not exist
# (the real id is "mmarco-mMiniLMv2"), so reranking was silently off.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Tokenizer: lowercase, strip punctuation, keep numbers whole, light stemming
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[a-z][a-z0-9]*")
STOPWORDS = frozenset(
    "a an and are as at be been by can did do does for from had has have how i if in into is it its "
    "me my of on or our over so than that the their them there these they this those to under up "
    "was we were what when where which while who whom why will with would you your".split()
)


def _stem(t: str) -> str:
    if len(t) <= 3 or t[0].isdigit():
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    for suffix in ("ment", "ing", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            t = t[: -len(suffix)]
            break
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Reranker: optional, fails loudly in logs but never takes search down
# ---------------------------------------------------------------------------
class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("BI_RERANKER", DEFAULT_RERANKER)
        self.model = None
        self.error: Optional[str] = None
        if self.model_name.lower() in ("", "none", "off"):
            self.error = "disabled by BI_RERANKER"
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker loaded: %s", self.model_name)
        except Exception as e:  # missing model, no network, low memory
            self.error = f"{type(e).__name__}: {e}"
            logger.error("Reranker %s unavailable, search will use fused ranking only: %s", self.model_name, e)

    @property
    def available(self) -> bool:
        return self.model is not None

    def scores(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.predict([[query, t] for t in texts]), dtype=np.float64)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class ReportIndex:
    def __init__(self, ids: List[str], docs: List[str], embeddings, pages: Dict[str, Optional[int]],
                 reranker: Optional[Reranker] = None):
        if not docs or len(docs) != len(embeddings) or len(ids) != len(docs):
            raise ValueError(f"Inconsistent index: {len(ids)} ids, {len(docs)} docs, {len(embeddings)} vectors")
        self.ids, self.docs = list(ids), list(docs)
        self.pages = [pages.get(i) for i in self.ids]
        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms
        self.bm25 = BM25Okapi([tokenize(d) or ["<empty>"] for d in self.docs])
        self.reranker = reranker

    @classmethod
    def load(cls, reranker: Optional[Reranker] = None) -> "ReportIndex":
        import chromadb

        col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION_NAME)
        r = col.get(include=["documents", "embeddings"])
        pages = {}
        if os.path.exists(PAGES_PATH):
            with open(PAGES_PATH, encoding="utf-8") as f:
                pages = json.load(f)
        logger.info("Loaded %d chunks, %d with page numbers", len(r["ids"]), sum(1 for v in pages.values() if v))
        return cls(r["ids"], r["documents"], r["embeddings"], pages, reranker)

    def search(self, query: str, query_vector=None, k: int = TOP_K, candidates: int = NUM_CANDIDATES) -> Dict:
        """Return {"passages": [...], "method": str, "reranked": bool}."""
        fused = np.zeros(len(self.docs))
        methods = []

        q_tokens = tokenize(query)
        if q_tokens:
            bm25 = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float64)
            if bm25.max() > 0:
                contrib = 1.0 / (RRF_K + _ranks(bm25))
                contrib[bm25 <= 0] = 0.0  # no keyword overlap, no keyword credit
                fused += contrib
                methods.append("BM25")

        q = np.asarray(query_vector, dtype=np.float32) if query_vector is not None else np.zeros(0)
        if q.size == self.embeddings.shape[1] and np.linalg.norm(q) > 0:
            fused += 1.0 / (RRF_K + _ranks(self.embeddings @ (q / np.linalg.norm(q))))
            methods.append("dense")

        cand = np.argsort(-fused, kind="stable")[:candidates]
        reranked = False
        if self.reranker is not None and self.reranker.available and len(cand):
            try:
                s = self.reranker.scores(query, [self.docs[i] for i in cand])
                cand = cand[np.argsort(-s, kind="stable")]
                reranked = True
            except Exception as e:
                logger.error("Reranking failed, using fused order: %s", e)

        top = cand[:k]
        return {
            "passages": [{"id": self.ids[i], "page": self.pages[i], "text": self.docs[i]} for i in top],
            "method": " + ".join(methods or ["none"]) + (" + rerank" if reranked else ""),
            "reranked": reranked,
        }
