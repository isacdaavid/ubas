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

# schiz_wholebrain

> A library to bride the gap between BIDS connectivity data and
> whole-brain simulation.

in progress...


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

Make sure to install the dependencies listed in [setup.cfg](https://github.com/isacdaavid/schiz_wholebrain/blob/master/setup.cfg#L50). For instance:

```{shell}
pip install numpy pybids scipy tqdm
```

If you wish to run the accompanying Jupyter notebook:
```{shell}
pip install notebook matplotlib
```
