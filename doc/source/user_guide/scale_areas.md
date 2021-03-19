scale_areas.py
==============

- Calculates area scaling factors for a polar stereographic projection

#### Calling Sequence
```python
from SMBcorr.scale_areas import scale_areas
scale = scale_areas(latitude, ref=70.0)
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/scale_areas.py)

#### Arguments
- `lat`: input latitude array (degrees North)

#### Keyword arguments
- `flat`: ellipsoidal flattening (default WGS84)
- `ref`: Standard parallel (latitude with no distortion, e.g. +70/-71)

#### Returns
- `scale`: area scaling factors

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
