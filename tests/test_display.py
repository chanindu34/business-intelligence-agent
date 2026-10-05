def test_model_maths_stays_literal_but_bold_still_works():
    from display import escape_md, plain_answer
    assert plain_answer("I cannot compute 9**9**9.") == r"I cannot compute 9\*\*9\*\*9."
    assert plain_answer("2 * 3 * 4 = 24") == r"2 \* 3 \* 4 = 24"
    assert plain_answer("**Retail** share is 40.67%") == "**Retail** share is 40.67%"
    assert plain_answer("costs $5") == r"costs \$5"
    once = plain_answer("9**9**9 costs $5")
    assert plain_answer(once) == once
    assert escape_md("9**9**9") == r"9\*\*9\*\*9"


def test_bold_numbers_are_left_alone():
    from display import plain_answer
    assert plain_answer("Retail was **40.67%** of the total") == "Retail was **40.67%** of the total"
    assert plain_answer("(80.01 - 45.85) * 100") == r"(80.01 - 45.85) \* 100"
