scale_factors.py
==============

- Calculates area scaling factors for a polar stereographic projection

#### Calling Sequence
```python
from SMBcorr.scale_factors import scale_factors
scale = scale_factors(latitude, reference_latitude=70.0)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/scale_factors.py)

#### Arguments
- `lat`: input latitude array (degrees North)

#### Keyword arguments
- `flat`: ellipsoidal flattening (default WGS84)
- `ref`: Standard parallel (latitude with no distortion, e.g. +70/-71)

#### Returns
- `scale`: area scaling factors

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
