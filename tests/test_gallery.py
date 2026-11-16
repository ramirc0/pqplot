import runpy
from pathlib import Path

import poikilos as pk
import pytest

SCRIPTS = sorted((Path(__file__).parents[1] / "gallery").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
@pytest.mark.mpl_image_compare(
    baseline_dir="baseline", style=pk.rc_params("zed"), savefig_kwargs={"dpi": 100}
)
def test_gallery(script):
    return runpy.run_path(str(script))["fig"]
