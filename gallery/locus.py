"""An ideogram over a locus, with the locus boxed and joined to the plot below."""

import matplotlib.pyplot as plt
import numpy as np

import pqplot as pqp

hg38 = pqp.Genome.bundled("hg38")
locus = pqp.Region.parse("chr17:43,044,295-43,125,483")
pos = np.linspace(locus.start, locus.end, 400)
signal = np.exp(-(((pos - 43.07e6) / 6e3) ** 2)) + 0.6 * np.exp(-(((pos - 43.11e6) / 4e3) ** 2))

fig, (head, body) = plt.subplots(2, 1, figsize=(6.5, 2.5), height_ratios=[1, 3])
ideo = pqp.ideogram(hg38[locus.chrom], ax=head)
body.fill_between(pos, signal, linewidth=0)
body.set(xlim=(locus.start, locus.end), ylabel="Signal")
body.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.2f} Mb")
ideo.zoom(locus, into=body)
