#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:19 2020

@author: ben
"""

import pointCollection as pc
import glob
import numpy as np
thedir='/Volumes/ice2/ben/scf/GL_11/007'

files=glob.glob(thedir+'/ATL11*.h5')
D_list=[None for ii in range(3*len(files))]
D_count=0
for file in files:
    if np.mod(files.index(file), 100)==0:
        print(files.index(file))
    for pair in [1, 2, 3]:
        try:
            D_list[D_count]=\
                          pc.ATL11.crossover_data(pair=pair).from_h5(file)
            D_count += 1
        except KeyError:
            print(f"error reading {file},  pair {pair}")

N_cols=D_list[0].shape[1]
D=pc.data(columns=N_cols).from_list(D_list[0:D_count])
D.get_xy(EPSG=3413)
print("Dumping dh to file")
D.to_h5(thedir+'/../U00_crossover_data_v1.h5')


# to get differences between early (pre-rgt) crossing tracks and later reference surfaces,
# the early index should be 1, the later index should be zero
dh=D.h_corr[:, 5, 0]-D.h_corr[:, 0,1]
good=np.isfinite(dh)
xx=np.zeros((dh.size,2))
yy=xx.copy()
xx[:, 0]=D.x[:, 0, 1]
xx[:, 1]=D.x[:, 5, 0]
yy[:, 0]=D.y[:, 0, 1]
yy[:, 1]=D.y[:, 5, 0]

xx=xx[np.isfinite(dh),:]
yy=yy[np.isfinite(dh),:]
dh=dh[np.isfinite(dh)]


