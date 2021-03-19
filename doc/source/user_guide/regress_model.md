regress_model.py
================

- Estimates a modeled time series for extrapolation by least-squares regression

#### Calling Sequence
```python
from SMBcorr.regress_model import regress_model
d_out = regress_model(t_in, d_in, t_out, ORDER=2,
    CYCLES=[0.25,0.5,1.0,2.0,4.0,5.0], RELATIVE=t_in[0])
```
[Source code](https://github.com/tsutterley/SMBcorr/blob/master/SMBcorr/regress_model.py)

#### Arguments
- `t_in`: input time array (year-decimal)
- `d_in`: input data array
- `t_out`: output time array for calculating modeled values (year-decimal)

#### Keyword arguments
- `ORDER`: polynomial order in design matrix (default quadratic)
- `CYCLES`: harmonic cycles in design matrix (year-decimal)
- `RELATIVE`: set polynomial fits relative to some value

#### Returns
- `d_out`: reconstructed time series

#### Dependencies
- [numpy: Scientific Computing Tools For Python](https://numpy.org)
