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

> Unified BIDS Analysis Scaler

![](https://raw.githubusercontent.com/isacdaavid/ubas/refs/heads/master/data/ubas_logo.svg)

Load and process BIDS datasets using modern, scalable and general-purpose
declarative interfaces, so that you can concentrate on the neuroscience.

The UBAS philosophy fits well with researchers who want control over their
heterogeneous analyses, but do not want to waste time writing slow and
error-prone loops to load and manipulate the data. In fact, UBAS implements _no_
neuroimaging analysis methods at all. Rather, it stands as an intermediary
interface between the data and the methods, and aims for compatibility with
existing algorithmic arsenals.

## Features

 - **BIDS transparency**: UBAS objects follow the same intuitive hierarchy as
   the BIDS data in the file system: cohorts containing subjects, containing
   sessions, containing modalities, containing files and metadata, etc.

 - **Data parallelism**: computation is accelerated by default using proven
   primitives from the big data field: collecting, storing, filtering and
   processing in parallel is as easy as passing a custom function defined for
   one element.

 - **Memory efficient**: Achieved via lazy evaluation. File contents are only
   loaded to RAM when needed. An in-memory cache is still provided to speed-up
   access.

 - **Seamless flexibility**: The same features are available across levels of
   the tree structure, whether cohorts, subjects, sessions, etc.

## Installation

```{shell}
 git clone https://github.com/isacdaavid/ubas.git
```

A conda environment is recommended. The repository is being developed
and tested with Python 3.9.

```{shell}
conda create --name myenv python=3.9

conda activate myenv
```

Make sure to install the dependencies listed in [setup.cfg](https://github.com/isacdaavid/ubas/blob/master/setup.cfg#L50):

```{shell}
pip install ...
```

If you wish to run the accompanying Jupyter [notebook](https://github.com/isacdaavid/ubas/blob/master/notebook.ipynb):
```{shell}
pip install notebook matplotlib importlib neurolib
```
