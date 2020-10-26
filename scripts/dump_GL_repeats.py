#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:25:40 2019

@author: ben
"""

import glob
import matplotlib.pyplot as plt
import ATL11
#from PointDatabase import mapData, point_data
import pointCollection as pc
import numpy as np
import h5py
import time

cycles=[3, 4, 5, 6, 7]
n_skip=4
t0=time.time()
if True:
    MOG=pc.grid.data().from_geotif('/Volumes/ice1/ben/MOG/2005/mog_2005_1km.tif')
    thedir='/Volumes/ice2/ben/scf/GL_11/007'
    files=glob.glob(thedir+'/ATL11*07.h5')
    xydh=[]
    data_count=0
    xydh=[None]* (3*len(files))
    for count, file in enumerate(files):
        if np.mod(count, 50)==0:
            print(f'{count} out of {len(files)}, dt={time.time()-t0}')
            t0=time.time()
        for pair in [1, 2, 3]:
            filePair=(file, pair)
            try:

                D11 = pc.ATL11.data().from_h5(file, pair=pair)

            except Exception as e:
                #print(file)
                #print(e)
                continue


            D11.get_xy(EPSG=3413)
            ind=np.arange(5, D11.x.shape[0], n_skip)
            els=(D11.x[ind,0] > MOG.x[0]) & (D11.x[ind,0] < MOG.x[-1]) & \
                (D11.y[ind,0] > MOG.y[0]) & (D11.y[ind,0] < MOG.y[-1])
            if els.size > 0:
                ind=ind[els]
                D11.assign({'file_ind':np.zeros_like(D11.x)+count})
                D11.index(ind)
                xydh[data_count]=D11
                data_count += 1

            #except:
            #    print("problem with "+file)

    D=pc.data(columns=D11.shape[1]).from_list(xydh)


D.to_h5('/Volumes/ice2/ben/scf/GL_11/rel007_dump_every_4th.h5')


if False:
    D3=D[np.arange(0, D.shape[0], 3)]

    gimp_mask=pc.grid.data().from_geotif('/Volumes/ice1/ben/GimpMasks_v1.1/GimpIceMask_1km.tif').interp(D3.x[:,0], D3.y[:,0])
    D3=D3[gimp_mask > 0.1]

    fig=plt.figure(4); plt.clf()
    xx=D3.x[:,0]
    yy=D3.y[:,0]
    hax=[]
    for col in np.arange(4):
        hax.append(fig.add_subplot(1,4,col+1))
        MOG.show(cmap='gray', vmin=14000, vmax=17000)
        dh=D3.h_corr[:,col+1]-D3.h_corr[:,col]
        ind=np.argsort(np.abs(dh))
        hl=plt.scatter(xx[ind], yy[ind], 1, marker='.', c=dh[ind], vmin=-0.5, vmax=0.5, cmap='Spectral')
        #hb=plt.colorbar(shrink=0.75, extend='both')
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        hax[-1].set_title(f'c{cycles[col+1]} minus c{cycles[col]}, m')
    fig.tight_layout()
    fig.colorbar(hl, ax=hax, location='bottom', shrink=0.25)
if False:
    plt.figure(1); xy0=plt.ginput()[0]; 
    ATL11_file=files[xydh.file_ind[np.argmin(np.abs(xydh.x+1j*xydh.y - (xy0[0]+1j*xy0[1])))].astype(int)]
    plt.figure(); 
    ATL11_multi_plot(ATL11_file, hemisphere=1)
    
    