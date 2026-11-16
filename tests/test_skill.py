import re
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

SKILL = Path(__file__).parents[1] / "skills" / "pqplot"
PAGES = sorted(SKILL.rglob("*.md"))
CODE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.stem)
def test_skill_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for block in CODE.findall(page.read_text()):
        exec(compile(block, str(page), "exec"), namespace)
    plt.close("all")
