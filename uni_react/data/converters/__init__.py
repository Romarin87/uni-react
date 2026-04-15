"""Data format converters – transform raw molecular datasets into HDF5.

All converters are runnable as standalone CLI tools::

    python -m uni_react.data.converters.cdft --help
    python -m uni_react.data.converters.gdb13 --help
    python -m uni_react.data.converters.inspect --help

Converters
----------
:mod:`cdft`                   – XYZ + CDFT text → HDF5
:mod:`check_cdft`             – Validate CDFT text files
:mod:`gdb13`                  – extXYZ (GDB13/GFN2-xTB) → HDF5
:mod:`ed`                     – ED tar.gz archives → HDF5
:mod:`reaction_triplets_h5`   – Extract reaction triplets from HDF5
:mod:`reaction_triplets_xyz`  – Extract reaction triplets from XYZ
:mod:`inspect`                – Inspect HDF5 file structure
"""
