import scipy.ndimage as snd
import scipy.interpolate as si
import numpy as np
import SMBcorr.utilities

# attempt imports
nc4 = SMBcorr.utilities.import_dependency('netCDF4')
pdb = SMBcorr.utilities.import_dependency('pdb')

def convert_delta_time(delta_time, gps_epoch=1198800018.0):
    # calculate gps time from delta_time
    gps_seconds = gps_epoch + delta_time
    time_leaps = SMBcorr.time.count_leap_seconds(gps_seconds)

    # calculate julian time
    julian = 2400000.5 + SMBcorr.time.convert_delta_time(gps_seconds - time_leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    # convert to calendar date
    Y,M,D,h,m,s = SMBcorr.time.convert_julian(julian,FORMAT='tuple')
    # calculate year-decimal time
    decimal = SMBcorr.time.convert_calendar_decimal(Y,M,day=D,
        hour=h,minute=m,second=s)
    # return both the Julian and year-decimal formatted dates
    return dict(julian=julian, decimal=decimal)
def make_zmelt_cumulative(nc_file):
    with nc4.Dataset(nc_file, 'r') as ds1:
        xx=np.array(ds1['x'])
        yy=np.array(ds1['y'])
        tt=np.array(ds1['time'])
        t_slices=np.flatnonzero(tt>2018)
        first_t_slice=np.min(t_slices)
        me=np.array(ds1['Me'][first_t_slice:,:,:])

    me[me<-1000]=np.NaN
    tt=tt[first_t_slice:]

    me_c=np.cumsum(me, axis=0)*(tt[1]-tt[0])
    # make a smoothed version of the cumulative melt
    me_cs=me_c.copy()
    me_cs[~np.isfinite(me_cs)]=0
    me_cs=snd.gaussian_filter(me_cs, sigma=[0, 1, 1])

    # make a mask variable, use it to correct for edge effects
    mask=np.isfinite(me_c)
    mask_s=snd.gaussian_filter(mask.astype(np.float64), sigma=[0, 1, 1])
    mask_s[mask_s==0]=np.NaN
    me_cs=me_cs/mask_s

    # write the finite portions of the unsmoothed grid on top of the smoothed grid
    me_cs[np.isfinite(me_c)] = me_c[np.isfinite(me_c)]

    # make interpolator objects for the smoothed, filled cumulative melt and mask
    meci=si.RegularGridInterpolator((tt, xx, yy), me_cs, method='linear')
    mai=si.RegularGridInterpolator((xx, yy), mask[0,:,:].astype(np.float64), method='linear')

    return meci, mai, tt

def interp_gsfc_zmelt(D, nc_file):
    meci, mai, t_vals = make_zmelt_cumulative(nc_file)
    for key in D:
        zm = np.zeros((D[key].x.size, 2))+np.NaN
        for col, field in enumerate(['t0', 't1']):
            tt = convert_delta_time(getattr(D[key], field))['decimal']
            good = (tt >= np.min(t_vals)) & (tt <= np.max(t_vals))
            zm[:, col][good] = meci.__call__((tt[good], D[key].x[good], D[key].y[good]))
        D[key].assign({'z_melt':zm[:,1]-zm[:,0]})
        mask_i = mai.__call__((D[key].x, D[key].y))
        D[key].z_melt[mask_i<0.1]=np.NaN

