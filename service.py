"""Shared, lazily built resources: one Gemini client, one index, one embedder.

Nothing touches the network or loads a model at import time, so tests and
tools can import any module cheaply.
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60


@lru_cache(maxsize=1)
def get_client():
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY (in .env locally, or as a secret when deployed).")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000))


@lru_cache(maxsize=1)
def get_index():
    from retrieval import ReportIndex, Reranker

    return ReportIndex.load(reranker=Reranker())


def get_embedder():
    from agent import make_embedder

    return make_embedder(get_client())
