import time

import pytest

from tools import calculate


@pytest.mark.parametrize("expr,expected", [
    ("47 * 12", "564"),
    ("10 / 4", "2.5"),
    ("(80.01 - 45.85) / 45.85 * 100", "74.50381679"),
    ("31.74 / 78.05 * 100", "40.66623959"),
    ("2 ** 10", "1024"),
    ("2^10", "1024"),               # caret treated as power, a common user habit
    ("1,250 + 750", "2000"),        # thousands separators accepted
    ("-5 + 3", "-2"),
    ("7 // 2", "3"),
    ("7 % 4", "3"),
])
def test_valid_arithmetic(expr, expected):
    r = calculate(expr)
    assert r["success"] and r["result"] == expected


@pytest.mark.parametrize("expr", ["9**9**9", "10**1000", "2**101", "(10**6+1)**2"])
def test_huge_powers_rejected_instantly(expr):
    # The first version computed 9**9**9 (~370 million digits) and froze the server.
    t0 = time.perf_counter()
    r = calculate(expr)
    assert not r["success"]
    assert time.perf_counter() - t0 < 0.1


@pytest.mark.parametrize("expr,error_part", [
    ("10 / 0", "division by zero"),
    ("5 +", "invalid syntax"),
    ("", "empty"),
    ("1" * 201, "longer than"),
    ("+".join(["1"] * 40), "too complex"),
    ("__import__('os').system('ls')", "unsupported syntax"),
    ("x + 1", "unsupported syntax"),
    ("True + 1", "unsupported syntax"),
    ("(-1) ** 0.5", "too large"),     # complex result is refused
])
def test_rejected_inputs(expr, error_part):
    r = calculate(expr)
    assert not r["success"] and error_part in r["error"]


def test_too_large_error_tells_model_not_to_compute():
    from tools import calculate
    out = calculate("9**9**9")
    assert out["success"] is False
    assert "Do not compute this yourself" in out["error"]
