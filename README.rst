.. image:: https://img.shields.io/github/workflow/status/NOC-MSM/cf-tools/CI?logo=github
    :target: https://github.com/NOC-MSM/cf-tools/actions
    :alt: GitHub Workflow CI Status

.. image:: https://codecov.io/gh/NOC-MSM/cf-tools/branch/main/graph/badge.svg?token=2DR8TODWUV
    :target: https://codecov.io/gh/NOC-MSM/cf-tools
    :alt: Code Coverage

.. image:: https://results.pre-commit.ci/badge/github/NOC-MSM/cf-tools/main.svg
    :target: https://results.pre-commit.ci/badge/github/NOC-MSM/cf-tools/main.svg
    :alt: pre-commit.ci status

cf-tools
========

Tools based on CF Conventions to post-process OGCMs.
The parent accessor gives access to general methods and properties. Child accessors inherit all methods and properties from the parent adding model-specific features.

Parent accessor: ``obj.cf_tools``

Child accessors:

* ``obj.nemo_tools``


NEMO quick start
----------------
.. code-block:: language

    # Import
    import xarray as xr
    from cf_tools.nemo import standardize_output
    from dask.distributed import Client
    from intake import open_catalog

    # Dask Client
    client = Client()

    # Select simulation from catalog
    intake_kwargs = dict(
        name="GSRIDGE36Z",
        time_freq="5d_mean",
    )
    cat = open_catalog(
        "https://raw.githubusercontent.com/NOC-MSM/intake-catalogs/master/catalog.yaml"
    )

    # Create a single standardized dataset with all grids
    domain = cat.NOCS.GSRIDGE(grid="mesh_mask", **intake_kwargs).to_dask()
    output = []
    for grid in ("T", "U", "V", "W", "ice"):
        ds = cat.NOCS.GSRIDGE(grid=grid, **intake_kwargs).to_dask()
        output += [standardize_output(ds, domain, hgrid="T" if grid == "ice" else None)]
    ds = xr.merge(output)

