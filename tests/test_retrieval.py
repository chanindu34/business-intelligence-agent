import numpy as np
import pytest

from retrieval import ReportIndex, tokenize

DOCS = [
    "Group EBITDA increased by 75% to Rs.80.01 billion",
    "Combined room inventory of 3,468 rooms under management",
    "Key risks include cybersecurity and macroeconomic volatility",
    "Supermarket expansion continued across the island",
]
IDS = ["c0", "c1", "c2", "c3"]
PAGES = {"c0": 14, "c1": 10, "c2": 141, "c3": 15}


class FakeReranker:
    available = True

    def scores(self, query, texts):
        return np.array([1.0 if "room" in t else 0.0 for t in texts])


def test_tokenizer_stems_and_strips():
    assert tokenize("manage") == tokenize("management")
    assert tokenize("EBITDA?") == ["ebitda"]
    assert tokenize("Rs.80.01 billion") == ["rs", "80.01", "billion"]


def test_keyword_search_returns_pages():
    idx = ReportIndex(IDS, DOCS, np.eye(4).tolist(), PAGES)
    out = idx.search("How many rooms under management?", None, k=1)
    assert out["passages"][0]["page"] == 10 and out["method"] == "BM25"


def test_dense_and_keyword_are_fused():
    idx = ReportIndex(IDS, DOCS, np.eye(4).tolist(), PAGES)
    out = idx.search("cybersecurity", [0, 0, 1, 0], k=1)
    assert out["passages"][0]["id"] == "c2" and out["method"] == "BM25 + dense"


def test_wrong_size_vector_falls_back_to_keywords():
    idx = ReportIndex(IDS, DOCS, np.eye(4).tolist(), PAGES)
    out = idx.search("supermarket", [1.0] * 7, k=1)
    assert out["passages"][0]["id"] == "c3" and "dense" not in out["method"]


def test_reranker_reorders():
    idx = ReportIndex(IDS, DOCS, np.eye(4).tolist(), PAGES, reranker=FakeReranker())
    out = idx.search("ebitda rooms", [1, 0, 0, 0], k=1)
    assert out["reranked"] and out["passages"][0]["id"] == "c1"


def test_inconsistent_index_is_rejected():
    with pytest.raises(ValueError):
        ReportIndex(IDS, DOCS, np.eye(3).tolist(), PAGES)
