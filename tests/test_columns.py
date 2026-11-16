import numpy as np
import pandas as pd
import polars as pl
import pytest

from pqplot.columns import column, rows_by_chrom

TABLE = {"chrom": ["chr2", "1", "chr1", "chrM"], "pos": [5, 7, 3, 9]}


@pytest.mark.parametrize("make", [dict, pd.DataFrame, pl.DataFrame])
def test_column_reads_any_table(make):
    np.testing.assert_array_equal(column(make(TABLE), "pos"), [5, 7, 3, 9])


def test_column_passes_values_through():
    np.testing.assert_array_equal(column(None, [1, 2]), [1, 2])


def test_column_by_name_needs_a_table():
    with pytest.raises(ValueError, match="table"):
        column(None, "pos")


def test_rows_group_by_chromosome_across_naming_styles_in_input_order():
    groups = rows_by_chrom(column(TABLE, "chrom"))
    assert {k: v.tolist() for k, v in groups.items()} == {"1": [1, 2], "2": [0], "MT": [3]}


@pytest.mark.parametrize("make", [pd.DataFrame, pl.DataFrame])
def test_rows_group_dataframe_strings(make):
    groups = rows_by_chrom(column(make(TABLE), "chrom"))
    assert sorted(groups) == ["1", "2", "MT"]


def test_rows_of_an_empty_table():
    assert rows_by_chrom(np.array([], dtype=str)) == {}
