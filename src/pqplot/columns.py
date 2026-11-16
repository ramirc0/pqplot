"""Column arrays from any table, checked and grouped by chromosome."""

from __future__ import annotations

import numpy as np

from pqplot.genome import chrom_key


def column(table: object, spec: object) -> np.ndarray:
    """Return one column of a table as a NumPy array.

    Parameters
    ----------
    table : DataFrame, dict or None
        Any table indexed by column name, such as a polars or pandas
        DataFrame, a pyarrow Table or a dict of sequences.
    spec : str or array-like
        Column name, or the values themselves.

    Returns
    -------
    numpy.ndarray
        At least one-dimensional.

    Raises
    ------
    ValueError
        If `spec` names a column but there is no table.
    KeyError
        If the table has no such column.

    Examples
    --------
    >>> column({"pos": [5, 7]}, "pos")
    array([5, 7])
    >>> column(None, [1.5])
    array([1.5])
    """
    if not isinstance(spec, str):
        return np.atleast_1d(np.asarray(spec))
    if table is None:
        raise ValueError(f"column {spec!r} needs a table; pass data or the values themselves")
    try:
        values = table[spec]
    except Exception as error:
        raise KeyError(f"no column {spec!r} in {type(table).__name__}") from error
    return np.atleast_1d(np.asarray(values))


def columns(table: object, **specs: object) -> dict[str, np.ndarray]:
    """Return the columns given by `specs`, skipping None, checked for one length.

    Raises
    ------
    ValueError
        If the columns differ in length.

    Examples
    --------
    >>> sorted(columns({"chrom": ["1"], "pos": [5]}, chrom="chrom", pos="pos", value=None))
    ['chrom', 'pos']
    """
    arrays = {name: column(table, spec) for name, spec in specs.items() if spec is not None}
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) > 1:
        sizes = ", ".join(f"{name} has {length}" for name, length in lengths.items())
        raise ValueError(f"columns differ in length: {sizes}")
    return arrays


def numbers(values: np.ndarray) -> np.ndarray:
    """Return values as floats, with nulls as NaN.

    Examples
    --------
    >>> numbers(np.array([1, None, "2.5"], dtype=object))
    array([1. , nan, 2.5])
    """
    if values.dtype != object:
        return values.astype(float)
    return np.array([_number(v) for v in values], dtype=float)


def _number(value: object) -> float:
    try:
        return float(value)
    except TypeError:
        return np.nan


def present(values: np.ndarray) -> np.ndarray:
    """Return which values are neither null nor NaN.

    Examples
    --------
    >>> present(np.array(["a", None, float("nan")], dtype=object)).tolist()
    [True, False, False]
    """
    if values.dtype.kind == "f":
        return ~np.isnan(values)
    if values.dtype != object:
        return np.ones(len(values), dtype=bool)
    return np.array([not _missing(v) for v in values], dtype=bool)


def _missing(value: object) -> bool:
    try:
        return value is None or bool(value != value)
    except TypeError:
        # pandas.NA refuses to become a bool.
        return True


def rows_by_chrom(chroms: np.ndarray, keep: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Group row indices by chromosome.

    Parameters
    ----------
    chroms : numpy.ndarray
        Chromosome name of each row. Names match across styles.
    keep : numpy.ndarray of bool, optional
        Rows to group. By default, all.

    Returns
    -------
    dict of str to numpy.ndarray
        Row indices in input order, keyed by `pqplot.genome.chrom_key`.

    Examples
    --------
    >>> {k: v.tolist() for k, v in rows_by_chrom(np.array(["chr2", "1", "chr1"])).items()}
    {'1': [1, 2], '2': [0]}
    """
    index = np.arange(len(chroms)) if keep is None else np.flatnonzero(keep)
    if len(index) == 0:
        return {}
    names, inverse = np.unique(chroms[index].astype(str), return_inverse=True)
    keys, key_index = np.unique([chrom_key(n) for n in names], return_inverse=True)
    rows = key_index[inverse]
    order = np.argsort(rows, kind="stable")
    groups = np.split(index[order], np.cumsum(np.bincount(rows, minlength=len(keys)))[:-1])
    return dict(zip(keys.tolist(), groups, strict=True))
