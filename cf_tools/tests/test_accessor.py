"""
Tests accessor for all models
"""
# pylint: disable=C0116

from cf_xarray.datasets import airds

import cf_tools  # noqa: F401 pylint: disable=W0611


def test_options():

    airds.cf_tools.set_options(a=1)
    assert airds.cf_tools.options["a"] == 1

    airds.cf_tools.unset_options("a")
    assert airds.cf_tools.options == {}

    with airds.cf_tools:
        airds.cf_tools.set_options(a=1)
        assert airds.cf_tools.options["a"] == 1
    assert airds.cf_tools.options == {}
