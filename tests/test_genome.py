import gzip

import numpy as np
import pytest

from pqplot import Band, Chromosome, Genome, Region, Stain
from pqplot.genome import chrom_key, natural_order

ASSEMBLIES = ["hg19", "hg38", "hs1", "mm10", "mm39"]


@pytest.fixture
def band_table(tmp_path):
    path = tmp_path / "cytoBand.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(
            "chr10\t0\t40\tq11\tgneg\n"
            "chr11\t0\t10\tpter\tgneg\n"
            "chr11\t10\t40\tqA1\tgpos33\n"
            "chr2\t60\t100\tq11\tgpos50\n"
            "chr2\t0\t40\tp11.2\tgneg\n"
            "chr2\t40\t50\tp11.1\tacen\n"
            "chr2\t50\t60\tq10\tacen\n"
            "chrX\t0\t30\tp11\tgneg\n"
            "chrM\t0\t16\t\t\n"
            "chr1_random\t0\t5\t\tgneg\n"
            "chrUn_gl1\t0\t5\t\tgneg\n"
            "chr7_alt\t0\t5\t\tgneg\n"
        )
    return path


def test_from_cytobands_keeps_primary_chromosomes_in_natural_order(band_table):
    assert [c.name for c in Genome.from_cytobands(band_table)] == ["chr2", "chr10", "chr11", "chrX"]


def test_from_cytobands_sorts_bands_and_measures_length(band_table):
    chr2 = Genome.from_cytobands(band_table)["chr2"]
    assert [b.start for b in chr2.bands] == [0, 40, 50, 60]
    assert chr2.length == 100
    assert chr2.bands[0] == Band(0, 40, "p11.2", Stain("gneg"))


def test_centromere_spans_acen_bands(band_table):
    assert Genome.from_cytobands(band_table)["chr2"].centromere == (40, 60)


def test_centromere_is_unknown_with_p_bands_but_no_acen(band_table):
    assert Genome.from_cytobands(band_table)["chrX"].centromere is None


@pytest.mark.parametrize("name", ["chr10", "chr11"])
def test_chromosome_without_numbered_p_bands_is_telocentric(band_table, name):
    chrom = Genome.from_cytobands(band_table)[name]
    assert chrom.centromere == (0, 0)
    assert chrom.telocentric


def test_from_cytobands_selects_chromosomes_in_given_order(band_table):
    genome = Genome.from_cytobands(band_table, chroms=["X", "chrM", "2"])
    assert [c.name for c in genome] == ["chrX", "chrM", "chr2"]


def test_from_cytobands_rejects_missing_chromosome(band_table):
    with pytest.raises(KeyError, match="chr3"):
        Genome.from_cytobands(band_table, chroms=["chr3"])


@pytest.mark.parametrize(
    ("stain", "density"),
    [
        ("gneg", 0.0),
        ("gpos25", 0.25),
        ("gpos33", 0.33),
        ("gpos66", 0.66),
        ("gpos100", 1.0),
        ("gpos", 1.0),
        ("acen", None),
        ("gvar", None),
        ("stalk", None),
    ],
)
def test_stain_density(stain, density):
    assert Stain(stain).density == density


@pytest.mark.parametrize(("a", "b"), [("chr1", "1"), ("chrM", "MT"), ("chrM", "M"), ("chrX", "X")])
def test_chrom_key_matches_naming_styles(a, b):
    assert chrom_key(a) == chrom_key(b)


def test_natural_order():
    names = ["chrY", "chr10", "chrM", "chrX", "chr2", "chr1", "chrW", "chrZ", "chrA"]
    assert sorted(names, key=natural_order) == [
        "chr1",
        "chr2",
        "chr10",
        "chrX",
        "chrY",
        "chrZ",
        "chrW",
        "chrM",
        "chrA",
    ]


def test_genome_lookup_matches_naming_styles(band_table):
    genome = Genome.from_cytobands(band_table)
    assert genome["2"] is genome["chr2"]
    assert "10" in genome
    assert "chr3" not in genome
    assert len(genome) == 4


def test_genome_lookup_of_missing_chromosome_raises(band_table):
    with pytest.raises(KeyError, match="chr3"):
        Genome.from_cytobands(band_table)["chr3"]


def test_autosomes(band_table):
    assert Genome.from_cytobands(band_table).autosomes == ["chr2", "chr10", "chr11"]


@pytest.mark.parametrize("name", ["x.chrom.sizes", "x.fa.fai"])
def test_from_sizes(tmp_path, name):
    path = tmp_path / name
    path.write_text("chr10\t500\t0\nchr1\t1000\t0\nchrM\t16\t0\nchr1_random\t5\t0\nchrX\t700\t0\n")
    genome = Genome.from_sizes(path)
    assert [(c.name, c.length) for c in genome] == [("chr1", 1000), ("chr10", 500), ("chrX", 700)]
    assert all(c.bands == () and c.centromere is None for c in genome)


def test_from_sizes_reads_centromere_bed(tmp_path):
    sizes = tmp_path / "x.chrom.sizes"
    sizes.write_text("chr1\t1000\nchr2\t500\n")
    bed = tmp_path / "centromeres.bed"
    bed.write_text("1\t400\t450\n1\t450\t520\n")
    genome = Genome.from_sizes(sizes, centromeres=bed)
    assert genome["chr1"].centromere == (400, 520)
    assert genome["chr2"].centromere is None


def test_from_sizes_selects_chromosomes(tmp_path):
    sizes = tmp_path / "x.chrom.sizes"
    sizes.write_text("chr1\t1000\nscaffold_7\t500\n")
    assert [c.name for c in Genome.from_sizes(sizes, chroms=["scaffold_7"])] == ["scaffold_7"]


@pytest.mark.parametrize("assembly", ASSEMBLIES)
def test_bundled_assemblies_hold_banded_primary_chromosomes(assembly):
    genome = Genome.bundled(assembly)
    autosomes = 19 if assembly.startswith("mm") else 22
    assert genome.autosomes == [f"chr{i}" for i in range(1, autosomes + 1)]
    assert [c.name for c in genome][autosomes:] == ["chrX", "chrY"]
    for chrom in genome:
        assert chrom.bands[0].start == 0
        assert all(a.end == b.start for a, b in zip(chrom.bands, chrom.bands[1:], strict=False))
        assert chrom.centromere is not None


@pytest.mark.parametrize(
    ("assembly", "length", "centromere"),
    [
        ("hg19", 249_250_621, (121_500_000, 128_900_000)),
        ("hg38", 248_956_422, (121_700_000, 125_100_000)),
        ("hs1", 248_387_328, (121_796_048, 126_300_487)),
    ],
)
def test_bundled_human_chr1(assembly, length, centromere):
    chr1 = Genome.bundled(assembly)["chr1"]
    assert (chr1.length, chr1.centromere) == (length, centromere)


@pytest.mark.parametrize("assembly", ["mm10", "mm39"])
def test_bundled_mouse_chromosomes_are_telocentric(assembly):
    assert all(c.telocentric for c in Genome.bundled(assembly))


def test_mouse_bands_have_intermediate_densities():
    densities = {b.stain.density for c in Genome.bundled("mm10") for b in c.bands}
    assert {0.33, 0.66} <= densities


def test_bundled_rejects_unknown_assembly():
    with pytest.raises(ValueError, match="hg38"):
        Genome.bundled("rheMac10")


def test_chromosome_without_centromere_is_not_telocentric():
    assert not Chromosome("chr1", 100).telocentric


@pytest.mark.parametrize(
    ("text", "region"),
    [
        ("chr17:43,000,000-43,200,000", Region("chr17", 43_000_000, 43_200_000)),
        (" 17:5-10 ", Region("17", 5, 10)),
        ("chr1_KI270706v1_random:1-100", Region("chr1_KI270706v1_random", 1, 100)),
    ],
)
def test_region_parse(text, region):
    assert Region.parse(text) == region


@pytest.mark.parametrize("text", ["chr17", "chr17:10", "chr17:20-10", "chr17:a-b"])
def test_region_parse_rejects_bad_text(text):
    with pytest.raises(ValueError, match="region"):
        Region.parse(text)


@pytest.fixture
def three():
    return Genome([Chromosome("chr1", 100), Chromosome("chr2", 50), Chromosome("chrX", 30)])


def test_concatenated_offsets(three):
    linear = three.concatenated(gap=10)
    assert [linear.offset(c) for c in ("chr1", "2", "X")] == [0, 110, 170]
    assert linear.length == 200


def test_concatenated_position_is_vectorized(three):
    linear = three.concatenated(gap=10)
    chroms = np.array(["chr2", "1", "chrX", "chr2", "chrM"])
    pos = np.array([5, 7, 0, 50, 3])
    np.testing.assert_array_equal(linear.position(chroms, pos), [115, 7, 170, 160, np.nan])


def test_concatenated_position_accepts_scalars(three):
    assert three.concatenated().position("chr2", 5) == 105


@pytest.mark.parametrize(("name", "label"), [("chr1", "1"), ("X", "X"), ("chrM", "M")])
def test_chromosome_label_drops_the_chr_prefix(name, label):
    assert Chromosome(name, 10).label == label


def sizes_file(tmp_path, text):
    path = tmp_path / "x.chrom.sizes"
    path.write_text(text)
    return path


def test_from_sizes_accepts_a_one_shot_iterator_of_chroms(tmp_path):
    path = sizes_file(tmp_path, "chr1\t100\nchr2\t50\n")
    assert [c.name for c in Genome.from_sizes(path, chroms=iter(["chr2"]))] == ["chr2"]


@pytest.mark.parametrize(
    "chromosomes",
    [
        [Chromosome("chr1", 100), Chromosome("1", 200)],
        [Chromosome("chrM", 16), Chromosome("MT", 16)],
    ],
)
def test_genome_rejects_chromosomes_that_share_a_name(chromosomes):
    with pytest.raises(ValueError, match="more than once"):
        Genome(chromosomes)


def test_from_sizes_rejects_a_repeated_name(tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        Genome.from_sizes(sizes_file(tmp_path, "chr1\t100\nchr1\t200\n"))


@pytest.mark.parametrize("text", ["", "scaffold_1\t100\nscaffold_2\t50\n", "Chr1\t100\n"])
def test_from_sizes_rejects_a_file_without_primary_chromosomes(tmp_path, text):
    with pytest.raises(ValueError, match="chroms="):
        Genome.from_sizes(sizes_file(tmp_path, text))


@pytest.mark.parametrize(
    ("length", "centromere"),
    [(0, None), (-5, None), (100, (30, 20)), (100, (-5, 10)), (100, (90, 500))],
)
def test_chromosome_rejects_impossible_lengths_and_centromeres(length, centromere):
    with pytest.raises(ValueError, match="chr1"):
        Chromosome("chr1", length, centromere=centromere)


def test_band_rejects_an_end_before_its_start():
    with pytest.raises(ValueError, match="band"):
        Band(10, 5, "p11", Stain("gneg"))


@pytest.mark.parametrize(("start", "end"), [(20, 10), (-5, 10)])
def test_region_rejects_impossible_spans(start, end):
    with pytest.raises(ValueError, match="region"):
        Region("chr1", start, end)


def test_region_parse_keeps_colons_in_the_name():
    assert Region.parse("HLA-A*01:01:1-100") == Region("HLA-A*01:01", 1, 100)


def test_region_parse_rejects_non_ascii_digits():
    with pytest.raises(ValueError, match="region"):
        Region.parse("chr1:١٢-٣٤")


def test_natural_order_treats_superscripts_as_names():
    assert natural_order("chr²") == (1, 5, "²")


@pytest.mark.parametrize("gap", [-1, float("nan"), float("inf")])
def test_concatenated_rejects_negative_or_infinite_gaps(three, gap):
    with pytest.raises(ValueError, match="gap"):
        three.concatenated(gap=gap)


def test_position_rejects_columns_of_different_lengths(three):
    with pytest.raises(ValueError, match="length"):
        three.concatenated().position(["chr2"], [1, 2, 3])


def test_position_is_nan_outside_the_chromosome(three):
    linear = three.concatenated(gap=10)
    np.testing.assert_array_equal(
        linear.position(["chr1", "chr1", "chr1", "chr2"], [-1, 0, 100, 51]),
        [np.nan, 0, 100, np.nan],
    )


def test_position_matches_float_chromosome_names(three):
    np.testing.assert_array_equal(three.concatenated().position([1.0, 2.0], [5, 5]), [5, 105])


def test_offset_of_a_missing_chromosome_names_it(three):
    with pytest.raises(KeyError, match="chr9"):
        three.concatenated().offset("chr9")
