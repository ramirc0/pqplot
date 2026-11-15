import gzip

import pytest

from karyogram_plot import cytobands


@pytest.fixture
def band_table(tmp_path):
    path = tmp_path / "cytoBand.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(
            "chr2\t100\t300\tp11\tgpos50\nchr2\t0\t100\tp12\tgneg\nchrUn_x\t0\t50\t\tgneg\n"
        )
    return path


def test_read_bands_groups_rows_by_chromosome(band_table):
    assert cytobands.read_bands(band_table) == {
        "chr2": [(100, 300, "p11", "gpos50"), (0, 100, "p12", "gneg")],
        "chrUn_x": [(0, 50, "", "gneg")],
    }


def test_read_bands_reads_plain_text(tmp_path):
    path = tmp_path / "bands.bed"
    path.write_text("chr1\t0\t10\tq11\tgneg\n")
    assert cytobands.read_bands(path) == {"chr1": [(0, 10, "q11", "gneg")]}


def test_read_sizes_reads_chrom_sizes_and_fai(tmp_path):
    sizes = tmp_path / "x.chrom.sizes"
    sizes.write_text("chr1\t1000\nchr2\t500\n")
    fai = tmp_path / "x.fa.fai"
    fai.write_text("chr1\t1000\t6\t60\t61\nchr2\t500\t1030\t60\t61\n")
    assert cytobands.read_sizes(sizes) == cytobands.read_sizes(fai) == {"chr1": 1000, "chr2": 500}


def test_read_sizes_skips_blank_and_comment_lines(tmp_path):
    path = tmp_path / "x.chrom.sizes"
    path.write_text("# header\nchr1\t1000\n\n")
    assert cytobands.read_sizes(path) == {"chr1": 1000}


def test_read_centromeres_merges_rows_into_one_span(tmp_path):
    path = tmp_path / "centromeres.bed"
    path.write_text("chr1\t120\t130\nchr1\t100\t110\tname\nchr2\t5\t9\n")
    assert cytobands.read_centromeres(path) == {"chr1": (100, 130), "chr2": (5, 9)}
