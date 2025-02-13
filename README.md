# slicer [alpha]
![License](https://img.shields.io/github/license/interpretml/slicer.svg?style=flat-square)
![Python Version](https://img.shields.io/pypi/pyversions/slicer.svg?style=flat-square)
![Package Version](https://img.shields.io/pypi/v/slicer.svg?style=flat-square)
![Maintenance](https://img.shields.io/maintenance/yes/2025.svg?style=flat-square)

*(Equal Contribution) Samuel Jenkins & Harsha Nori & Scott Lundberg*

**slicer** wraps tensor-like objects and provides a uniform slicing interface via `__getitem__`.

<br/>
It supports many data types including:

&nbsp;&nbsp;
[numpy](https://github.com/numpy/numpy) |
[pandas](https://github.com/pandas-dev/pandas) |
[scipy](https://docs.scipy.org/doc/scipy/reference/sparse.html) |
[pytorch](https://github.com/pytorch/pytorch) |
[list](https://github.com/python/cpython) |
[tuple](https://github.com/python/cpython) |
[dict](https://github.com/python/cpython)

And enables upgraded slicing functionality on its objects:
```python
# Handles non-integer indexes for slicing.
S(df)[:, ["Age", "Income"]]

# Handles nested slicing in one call.
S(nested_list)[..., :5]
```

It can also simultaneously slice many objects at once:
```python
# Gets first elements of both objects.
S(first=df, second=ar)[0, :]
```

This package has **0** dependencies. Not even one.

## Installation

Python 3.6+ | Linux, Mac, Windows
```sh
pip install slicer
```

## Getting Started

Basic anonymous slicing:
```python
from slicer import Slicer as S
li = [[1, 2, 3], [4, 5, 6]]
S(li)[:, 0:2].o
# [[1, 2], [4, 5]]
di = {'x': [1, 2, 3], 'y': [4, 5, 6]}
S(di)[:, 0:2].o
# {'x': [1, 2], 'y': [4, 5]}
```

Basic named slicing:
```python
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 3], 'B': [2, 4]})
ar = np.array([[5, 6], [7, 8]])
sliced = S(first=df, second=ar)[0, :]
sliced.first
# A    1
# B    2
# Name: 0, dtype: int64
sliced.second
# array([5, 6])
```

Real example:
```python
from slicer import Slicer as S
from slicer import Alias as A

data = [[1, 2], [3, 4]]
values = [[5, 6], [7, 8]]
identifiers = ["id1", "id1"]
instance_names = ["r1", "r2"]
feature_names = ["f1", "f2"]
full_name = "A"

slicer = S(
    data=data,
    values=values,
    # Aliases are objects that also function as slicing keys.
    # A(obj, dim) where dim informs what dimension it can be sliced on.
    identifiers=A(identifiers, 0),
    instance_names=A(instance_names, 0),
    feature_names=A(feature_names, 1),
    full_name=full_name,
)

sliced = slicer[:, 1]  # Tensor-like parallel slicing on all objects
assert sliced.data == [2, 4]
assert sliced.instance_names == ["r1", "r2"]
assert sliced.feature_names == "f2"
assert sliced.values == [6, 8]

sliced = slicer["r1", "f2"]  # Example use of aliasing
assert sliced.data == 2
assert sliced.feature_names == "f2"
assert sliced.instance_names == "r1"
assert sliced.values == 6
```

## Contact us
Raise an issue on GitHub, or contact us at interpret@microsoft.com
