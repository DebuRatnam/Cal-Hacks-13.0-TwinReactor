"""Regression: reproduce published 2-group benchmark k_eff with ndiffusion before generating data."""

import pytest

nd = pytest.importorskip("ndiffusion")


@pytest.mark.skip(reason="TODO: wire up ndiffusion's built-in IAEA 2-D benchmark")
def test_iaea_2d_keff():
    # Published IAEA 2-D reference k_eff ~1.02959. Check against the value shipped with ndiffusion.
    ...


@pytest.mark.skip(reason="TODO: wire up ndiffusion's built-in BIBLIS benchmark")
def test_biblis_keff():
    ...
