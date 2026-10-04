import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")


@pytest.fixture(autouse=True)
def _clear_model_cooldowns():
    import agent
    agent._benched.clear()
    yield
    agent._benched.clear()
