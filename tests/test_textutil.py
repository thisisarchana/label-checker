from rag.make_question_set import validate
from rag.textutil import content_words, longest_shared_run, words


def test_longest_shared_run_counts_consecutive_words_only():
    a = words("the food must contain twenty percent of the daily value")
    b = words("a food that must contain twenty percent is high")
    assert longest_shared_run(a, b) == 4   # "must contain twenty percent"


def test_content_words_drop_short_and_common_words():
    assert content_words("What is the limit for gluten in a food?") == {"limit", "gluten", "food"}


SOURCE = ("Gluten-free foods must contain less than 20 parts per million gluten. "
          "Compliance is judged by a scientifically valid analytical method.")


def row(question):
    return {"chunk_id": "c1", "section": "101.91", "citation": "21 CFR 101.91(b)", "question": question}


def test_validate_keeps_a_good_question_and_drops_bad_ones():
    rows = [
        row("How much gluten can a product contain before it cannot be called gluten-free?"),
        row("What limit applies?"),
        row("What does section 101.91 say about gluten limits for products?"),
        row("Foods must contain less than 20 parts per million gluten according to the rules?"),
        row("Which cooking temperatures are best for baking frozen pizza at home?"),
        row("How much gluten can a product contain before it cannot be called gluten-free here?"),
        row("When does using the word on a food package skip the usual restrictions on gluten?"),
    ]
    kept, dropped, rejected = validate(rows, {"c1": SOURCE})
    assert [r["question"] for r in kept] == [rows[0]["question"]]
    assert dropped == {"too short": 1, "mentions the source": 1, "copies the source text": 1,
                       "unrelated to the source": 1, "near-duplicate": 1, "vague reference": 1}
    assert {r["reason"] for r in rejected} == set(dropped)


def test_a_heavily_paraphrased_question_is_kept():
    source = "Calcium intake may reduce the risk of osteoporosis, a condition of reduced bone mass."
    kept, dropped, _ = validate(
        [row("Which approved phrasing lets a manufacturer talk about the link between calcium and bone thinning?")],
        {"c1": source},
    )
    assert len(kept) == 1 and not dropped