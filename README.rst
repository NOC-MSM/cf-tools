.. image:: https://img.shields.io/github/workflow/status/NOC-MSM/cf-tools/CI?logo=github
    :target: https://github.com/NOC-MSM/cf-tools/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/codecov/c/github/NOC-MSM/cf-tools.svg
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
