import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import poikilos as pk  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def zed():
    with pk.context("zed"):
        yield
    plt.close("all")
