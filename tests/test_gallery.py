import runpy
from pathlib import Path

import matplotlib as mpl
import poikilos as pk
import pytest

SCRIPTS = sorted((Path(__file__).parents[1] / "gallery").glob("*.py"))
MATPLOTLIB = "3.11"
# Bundled fonts, so the images do not depend on what the machine has installed.
STYLE = pk.rc_params("zed") | {"font.sans-serif": ["DejaVu Sans"]}

pytestmark = [
    pytest.mark.images,
    pytest.mark.skipif(
        not mpl.__version__.startswith(f"{MATPLOTLIB}."),
        reason=f"baselines are for matplotlib {MATPLOTLIB}",
    ),
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
@pytest.mark.mpl_image_compare(baseline_dir="baseline", style=STYLE, savefig_kwargs={"dpi": 300})
def test_gallery(script):
    return runpy.run_path(str(script))["fig"]
