import xml.etree.ElementTree as ET

from rag.chunk_cfr import MAX_CHARS, blocks, build_chunks, label_blocks, level_of, split_long


def test_levels_of_markers():
    assert level_of("a", None) == 1
    assert level_of("1", "b") == 2
    assert level_of("A", "b") == 4
    assert level_of("ii", "b") == 3


def test_lone_i_is_level_one_only_after_h():
    assert level_of("i", "h") == 1
    assert level_of("i", "b") == 3
    assert level_of("v", "u") == 1
    assert level_of("v", "c") == 3


def test_short_section_gives_one_chunk_per_lettered_paragraph():
    texts = ["(a) General.", "(1) First.", "(2) Second.", "(b) High claims.", "(1) Terms.", "(i) Sub item."]
    chunks = build_chunks("101.99", label_blocks(texts))
    assert [citation for citation, _ in chunks] == ["101.99(a)", "101.99(b)"]
    assert "(2) Second." in chunks[0][1]


def test_text_before_the_first_letter_gets_the_bare_section_citation():
    chunks = build_chunks("101.1", label_blocks(["Intro text.", "(a) First."]))
    assert [citation for citation, _ in chunks] == ["101.1", "101.1(a)"]


def test_long_paragraph_splits_at_numbered_subparagraphs_and_repeats_the_lead_in():
    texts = ["(b) The following apply:"] + [f"({n}) " + "word " * 140 for n in (1, 2, 3)]
    chunks = build_chunks("101.54", label_blocks(texts))
    assert [citation for citation, _ in chunks] == ["101.54(b)(1)", "101.54(b)(2)", "101.54(b)(3)"]
    assert all(text.startswith("(b) The following apply:") for _, text in chunks)
    assert all(len(text) <= MAX_CHARS for _, text in chunks)


def test_no_chunk_exceeds_the_limit_even_for_one_huge_table():
    table = "\n".join("row | " + "x" * 200 for _ in range(40))
    chunks = build_chunks("101.9", label_blocks(["(a) Intro.", table]))
    assert chunks and all(len(text) <= MAX_CHARS for _, text in chunks)


def test_split_long_handles_a_line_with_no_breaks():
    assert all(len(p) <= MAX_CHARS for p in split_long("y" * (MAX_CHARS * 2 + 5)))


def test_blocks_flattens_tables_and_skips_citation_notes():
    section = ET.fromstring(
        "<DIV8><HEAD>head</HEAD><P>(a) Text <I>italic</I> end.</P>"
        "<EXTRACT><P>(1) Inside extract.</P></EXTRACT>"
        "<TABLE><TR><TH>Name</TH><TH>Value</TH></TR><TR><TD>Fat</TD><TD>65 g</TD></TR></TABLE>"
        "<CITA>[58 FR 2431]</CITA></DIV8>"
    )
    assert list(blocks(section)) == [
        "(a) Text italic end.", "(1) Inside extract.", "Name | Value\nFat | 65 g",
    ]