"""Text clean-up before Streamlit renders it as markdown."""

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$])")
# '*' or '**' with a number or ')' just before (spaces allowed) and a number or '(' after.
_ARITH_STARS = re.compile(r"(?<=[\d)])(\s*)(\*{1,2})(?=\s*[\d(])")


def escape_md(text: str) -> str:
    """Show user text literally: '9**9**9' must not render as a bold '999'."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def plain_answer(text: str) -> str:
    """Keep model maths literal while leaving its bold and lists alone.

    '$...$' would render as LaTeX, and '*' or '**' between numbers ('9**9**9',
    '2 * 3 * 4') would turn into bold or italics. Safe to apply twice.
    """
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    return _ARITH_STARS.sub(lambda m: m.group(1) + "\\*" * len(m.group(2)), text)
