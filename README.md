<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/schiz_wholebrain.svg?branch=main)](https://cirrus-ci.com/github/<USER>/schiz_wholebrain)
[![ReadTheDocs](https://readthedocs.org/projects/schiz_wholebrain/badge/?version=latest)](https://schiz_wholebrain.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/schiz_wholebrain/main.svg)](https://coveralls.io/r/<USER>/schiz_wholebrain)
[![PyPI-Server](https://img.shields.io/pypi/v/schiz_wholebrain.svg)](https://pypi.org/project/schiz_wholebrain/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/schiz_wholebrain.svg)](https://anaconda.org/conda-forge/schiz_wholebrain)
[![Monthly Downloads](https://pepy.tech/badge/schiz_wholebrain/month)](https://pepy.tech/project/schiz_wholebrain)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/schiz_wholebrain)
-->

# UBAS

> Unified BIDS Analysis Suite

UBAS allows you to read and process BIDS datasets using modern,
scalable and general-purpose declarative interfaces, so that you can
concentrate on the neuroscience.

The BIDS standard is being increasingly adopted to store and
preprocess all sorts of neuroimaging data. However, flexible and
efficient tools to help researchers implement their in-house studies
starting from a BIDS folder are lacking. UBAS builds on PyBIDS, which
provides primitives to load and traverse BIDS dataset metadata, and
adds the capacity to represent and analyze the actual neuroimaging
data in a dependable and transparent structure which mimics the BIDS
hierarchy.

Instead of shipping prefabricated analysis pipelines for common use cases (like
Clinica), the UBAS philosophy fits with researchers who want control over their
heterogeneous analyses, but do not want to waste time writing error-prone loops.
It also appeals to researchers who would like to further standardize adoption of
BIDS, while exploiting UBAS scalability capabilities as a computational engine.

## Features

 - BIDS transparency: once loaded, UBAS objects follow the same
   file system hierarchy: cohorts containing subjects, containing
   sessions, containing modalities, containing files and metadata,
   etc.

 - Lazy evaluation: Datasets can be huge. File contents are only
   loaded to RAM when needed. An in-memory cache is
   still provided to speed-up access to popular files.

 - Parallel computing by default: accelerate the science with proven
   functional programming primitives used in big data. Want to filter
   your dataset or compute something on every member at certain level
   of your BIDS hierarchy? It's as easy as passing your custom
   function for one element. UBAS will automatically distribute the
   computation to the whole dataset.


## Installation

```{shell}
 git clone https://github.com/isacdaavid/schiz_wholebrain.git
```

A conda environment is recommended. The repository is being developed
and tested with Python 3.9.

```{shell}
conda create --name myenv python=3.9

conda activate myenv
```

Make sure to install the dependencies listed in [setup.cfg](https://github.com/isacdaavid/schiz_wholebrain/blob/master/setup.cfg#L50):

```{shell}
pip install numpy pybids scipy tqdm neurolib
```

If you wish to run the accompanying Jupyter [notebook](https://github.com/isacdaavid/schiz_wholebrain/blob/master/notebook.ipynb):
```{shell}
pip install notebook matplotlib importlib
```
