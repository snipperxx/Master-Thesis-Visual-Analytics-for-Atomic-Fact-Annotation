"""Regression guards for the OJ flattened-table reconstruction (src/oj_table.py).

Context: gold decision log D-01. The OJ plain-text export encodes a table as
one line, `1.2.3.4 // cell // cell ...`, where the dotted marker declares the
column count and every `//` introduces a cell — empties included, because the
empties are how a vertically merged cell is represented.

The guards that matter here are the ones that stop the reconstruction from
being confidently wrong: a dotted cross-reference in prose must not be read as
a table, a table whose geometry cannot be recovered must say so rather than
invent rows, and forward-filling a merged cell must not spill into columns the
merge never covered.
"""

from src.oj_table import (
    find_flattened_tables,
    normalize_flattened_tables,
    render,
    to_annex_table,
)
from src.table_linearize import linearize_table

def _encode(grid: list[list[str]]) -> str:
    """Render a grid the way the OJ plain-text export does.

    Stating fixtures as a grid and encoding them here keeps the tests honest:
    hand-writing the `//` runs makes it far too easy to be one separator out,
    which silently turns the fixture ragged and tests nothing.
    """
    n_cols = len(grid[0])
    assert all(len(row) == n_cols for row in grid), "grid must be rectangular"
    marker = ".".join(str(i + 1) for i in range(n_cols))
    cells = [c for row in grid for c in row]
    return marker + "".join(f" // {c}" for c in cells)


_BLANK4 = ["", "", "", ""]

# Shape of train-000121 article 1: real headers, two data rows, blank padding.
WITH_HEADERS = _encode([
    _BLANK4,
    ["Order No", "Category (unit)", "CN code", "Description"],
    _BLANK4,
    _BLANK4,
    ["40.0260", "26 (1 000 pieces)", "6104 41 00", "Women's dresses"],
    ["40.0290", "29 (1 000 pieces)", "6204 11 00", "Women's suits"],
    _BLANK4,
])

# Shape of train-000019 article 1: numeric column labels, plus a two-column
# vertical merge ("'U" / the authority) spanning all four data rows. The "(b)"
# in the last column belongs to exactly one row.
POSITIONAL_MERGED = _encode([
    _BLANK4,
    ["1", "2", "3", "4"],
    _BLANK4,
    ["'U", "Ministerio de Ganaderia", "- 66/401 Festuca", ""],
    ["", "", "- 66/402 Sorghum bicolor", "(b)"],
    ["", "", "Zea mays", ""],
    ["", "", "- 69/208 Helianthus annuus'", ""],
])


def test_headers_and_rows_recovered():
    (ft,) = find_flattened_tables(WITH_HEADERS)
    assert ft.n_cols == 4
    assert ft.ragged is False
    assert ft.headers == ["Order No", "Category (unit)", "CN code", "Description"]
    assert len(ft.rows) == 2
    assert ft.rows[0][0] == "40.0260"
    assert ft.rows[1][3] == "Women's suits"


def test_empty_cells_are_kept_so_the_grid_survives():
    """Dropping empties would renumber the columns and silently misalign rows."""
    (ft,) = find_flattened_tables(WITH_HEADERS)
    assert len(ft.cells) % ft.n_cols == 0
    assert len(ft.cells) > ft.n_data_cells


def test_numeric_column_labels_are_not_treated_as_headers():
    """"1 // 2 // 3 // 4" is a layout placeholder, not a set of labels.

    Reattaching them would emit "1 = 'U", which reads as a fact about the
    number 1.
    """
    (ft,) = find_flattened_tables(POSITIONAL_MERGED)
    assert ft.headers == []
    assert all(not c.isdigit() for row in ft.rows for c in row if c)


def test_vertical_merge_is_carried_down_but_only_the_leading_run():
    (ft,) = find_flattened_tables(POSITIONAL_MERGED)
    assert ft.merged_fill is True
    # Every row now names the country and authority the merge covered, so each
    # row is independently interpretable.
    assert all(row[0] == "'U" for row in ft.rows)
    assert all(row[1] == "Ministerio de Ganaderia" for row in ft.rows)
    # "(b)" sits in the last column of one row only. Forward-filling interior
    # or trailing gaps would attach it to rows it never applied to.
    last_col = [row[3] for row in ft.rows]
    assert last_col.count("(b)") == 1


def test_ragged_table_is_flagged_and_no_rows_are_invented():
    ragged = "1.2.3 // a // b // c // d"          # 4 cells, 3 columns
    (ft,) = find_flattened_tables(ragged)
    assert ft.ragged is True
    assert ft.rows == []
    text = render(ft)
    assert "could not be reconstructed" in text
    assert "merged or spanning cells" in text
    # The content still reaches the reader, just without a fabricated geometry.
    for cell in ("a", "b", "c", "d"):
        assert cell in text


def test_dotted_cross_references_in_prose_are_not_tables():
    """Legal prose is full of dotted references; none of them start a table."""
    prose = (
        "In Part I (1), point 1.1, the entry 'U = Uruguay' is inserted.\n"
        "Having regard to Article 18(5) thereof, and to point 2.3.4 below.\n"
        "See also 1.2.3.4 of the notice, which is not a table.\n"
    )
    assert find_flattened_tables(prose) == []


def test_marker_must_be_consecutive_from_one():
    assert find_flattened_tables("2.3.4 // a // b // c") == []
    assert find_flattened_tables("1.3.4 // a // b // c") == []
    assert len(find_flattened_tables("1.2.3 // a // b // c")) == 1


def test_render_does_not_claim_an_annex_location():
    """These tables sit inline in an article; "In Annex ..." would be false."""
    (ft,) = find_flattened_tables(WITH_HEADERS)
    text = render(ft)
    assert "Annex" not in text
    assert "set out in this Article" in text


def test_default_annex_prefix_is_unchanged_for_the_html_path():
    """scripts/run_dry_run.py relies on the original wording."""
    (ft,) = find_flattened_tables(WITH_HEADERS)
    at = to_annex_table(ft, annex_id="I", table_id=2)
    rows = linearize_table(at)
    assert rows[0].sentence.startswith("In Annex I, Table 2, row 0: ")


def test_normalize_replaces_the_run_and_keeps_surrounding_text():
    text = (
        "Article 1\n"
        "The following products are listed:\n"
        + WITH_HEADERS + "\n"
        "Article 2\n"
        "This Regulation shall enter into force.\n"
    )
    out, tables = normalize_flattened_tables(text)
    assert len(tables) == 1
    assert "//" not in out
    assert "1.2.3.4" not in out
    # Structural landmarks must survive, or article slicing breaks.
    assert "Article 1\n" in out
    assert "\nArticle 2\n" in out
    assert "This Regulation shall enter into force." in out
    assert "Order No = 40.0260" in out


def test_normalize_is_a_no_op_without_tables():
    text = "Article 1\nNothing tabular here.\n"
    out, tables = normalize_flattened_tables(text)
    assert out == text
    assert tables == []


def test_multiple_tables_keep_document_order():
    text = WITH_HEADERS + "\nArticle 2\n" + POSITIONAL_MERGED + "\n"
    out, tables = normalize_flattened_tables(text)
    assert len(tables) == 2
    assert tables[0].n_cols == 4 and tables[0].headers
    assert tables[1].n_cols == 4 and not tables[1].headers
    assert out.index("Order No = 40.0260") < out.index("Ministerio de Ganaderia")
