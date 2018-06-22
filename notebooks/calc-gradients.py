
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-ECCO-data" data-toc-modified-id="Read-ECCO-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read ECCO data</a></span><ul class="toc-item"><li><span><a href="#Do-monthly-with-dask" data-toc-modified-id="Do-monthly-with-dask-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Do monthly with dask</a></span></li><li><span><a href="#Compare-annual-and-monthly" data-toc-modified-id="Compare-annual-and-monthly-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Compare annual and monthly</a></span></li></ul></li><li><span><a href="#Read-Argo-climatology" data-toc-modified-id="Read-Argo-climatology-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Read Argo climatology</a></span><ul class="toc-item"><li><span><a href="#Do-monthly-with-dask" data-toc-modified-id="Do-monthly-with-dask-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Do monthly with dask</a></span></li></ul></li><li><span><a href="#Compare-ECCO-and-Argo-gradients?" data-toc-modified-id="Compare-ECCO-and-Argo-gradients?-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Compare ECCO and Argo gradients?</a></span><ul class="toc-item"><li><span><a href="#Histograms" data-toc-modified-id="Histograms-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Histograms</a></span></li><li><span><a href="#smooth-argo" data-toc-modified-id="smooth-argo-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>smooth argo</a></span></li></ul></li></ul></div>

# # Read ECCO data

# In[5]:


ρfile = '../datasets/ecco/interp_climatology/RHOAnoma.0001.nc'
Sfile = '../datasets/ecco/interp_climatology/SALT.0001.nc'
Tfile = '../datasets/ecco/interp_climatology/THETA.0001.nc'

T = xr.open_dataset(Tfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco)
S = xr.open_dataset(Sfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco)
ρ = xr.open_dataset(ρfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco)

ecc = xr.merge([T, S, ρ]).rename({'THETA': 'Tmean', 'SALT': 'Smean'})

# roll so that pacific is in the middle and we have coverage of all 3 basins
ecc = dcpy.oceans.dataset_center_pacific(ecc)

annual = ecc.mean(dim='time')
annual['ρmean'] = annual.RHOAnoma + 1029 # IN SITU DENSITY ANOMALY!!!!

ed.estimate_clim_gradients(annual)

annual.attrs['name'] = "Mean fields and isopycnal, diapycnal gradients from ECCO v4r3"
annual.attrs['dataset'] = 'ecco'
annual.to_netcdf('../datasets/ecco_annual_iso_gradient.nc', compute=True)

annual


# ##  Do monthly with dask

# In[1]:


ρfile = '../datasets/ecco/interp_climatology/RHOAnoma.0001.nc'
Sfile = '../datasets/ecco/interp_climatology/SALT.0001.nc'
Tfile = '../datasets/ecco/interp_climatology/THETA.0001.nc'

chunks = {'time': 12}

T = xr.open_dataset(Tfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco).chunk(chunks)
S = xr.open_dataset(Sfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco).chunk(chunks)
ρ = xr.open_dataset(ρfile, autoclose=True,
                    decode_times=False).pipe(ed.format_ecco).chunk(chunks)

ecc = xr.merge([T, S, ρ]).rename({'THETA': 'Tmean', 'SALT': 'Smean'})

# roll so that pacific is in the middle and we have coverage of all 3 basins
ecc = dcpy.oceans.dataset_center_pacific(ecc)

ecc


# In[2]:


monthly = ecc
monthly['ρmean'] = monthly.RHOAnoma + 1029

ed.estimate_clim_gradients(monthly)

monthly.attrs['name'] = "Mean fields and isopycnal, diapycnal gradients from ECCO v4r3"
monthly.attrs['dataset'] = 'ecco'

from dask.diagnostics import ProgressBar

delayed = monthly.to_netcdf('../datasets/ecco_iso_gradient.nc', compute=False)

with ProgressBar():
    results = delayed.compute(num_workers=2)
    
results


# ## Compare annual and monthly

# In[6]:


def plot_grad(input, ax):
    (np.log10(input.dTiso).sel(lat=slice(5, 18), lon=slice(82, 92), pres=slice(0, 200))
     .plot.hist(ax=ax[0], bins=20, density=True, histtype='step'))
    
    (np.log10(input.dTdz).sel(lat=slice(5, 18), lon=slice(82, 92), pres=slice(0, 200))
     .plot.hist(ax=ax[1], bins=20, density=True, histtype='step'))
    
f, ax = plt.subplots(1,2, sharex=True, sharey=True)
plot_grad(annual, ax)
plot_grad(monthly, ax)
plt.legend(('annual mean', 'dec'))


# In[8]:


annual.Tmean.sel(pres=0, method='nearest').plot()


# # Read Argo climatology

# In[7]:


argoT = xr.open_dataset('../datasets/argoclim/RG_ArgoClim_Temperature_2016.nc',
                        autoclose=True, decode_times=False)
argoS = xr.open_dataset('../datasets/argoclim/RG_ArgoClim_Salinity_2016.nc',
                        autoclose=True, decode_times=False)

argo = xr.merge([argoT, argoS])

argo = (argo.rename({'ARGO_TEMPERATURE_MEAN': 'Tmean',
                    'ARGO_TEMPERATURE_ANOMALY': 'Tanom',
                    'ARGO_SALINITY_MEAN': 'Smean',
                    'ARGO_SALINITY_ANOMALY': 'Sanom',
                    'LATITUDE': 'lat',
                    'LONGITUDE': 'lon',
                    'PRESSURE': 'pres',
                    'TIME': 'time'})
        .drop(['BATHYMETRY_MASK', 'Tanom', 'Sanom']))

_, pmat = xr.broadcast(argo.Tmean, argo.pres)

argo['ρmean'] = xr.DataArray(sw.dens(argo.Smean, argo.Tmean, pmat),
                             dims=argo.Smean.dims, coords=argo.Smean.coords)

del pmat

argo_annual = argo.mean(dim='time')

# apply an averaging operator?
ed.estimate_clim_gradients(argo_annual)

argo_annual.attrs['name'] = 'Mean fields, anomalies and isopycnal, diapycnal gradients from Argo climatology'
argo_annual.attrs['dataset'] = 'argo'
argo_annual.to_netcdf('../datasets/argo_annual_iso_gradients.nc')

argo_annual


# ## Do monthly with dask

# In[9]:


monthly = argo

ed.estimate_clim_gradients(monthly)

monthly.attrs['name'] = "Mean fields and isopycnal, diapycnal gradients from Argo"
monthly.attrs['dataset'] = 'argo'

from dask.diagnostics import ProgressBar

delayed = monthly.to_netcdf('../datasets/argo_monthly_iso_gradient.nc', compute=False)

with ProgressBar():
    results = delayed.compute(num_workers=2)
    
results


# # Compare ECCO and Argo gradients?

# In[12]:


def plot_gradients(input, region):
        
    isokwargs = dict(yincrease=False, cmap=mpl.cm.RdYlBu_r, center=0, robust=True)
    ρkwargs = dict(yincrease=False, levels=50, colors='w')
    TSkwargs = dict(yincrease=False, levels=40, colors='k')

    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)

    f.set_size_inches(12, 8)

    (input.dTiso.sel(**region)
     .plot(ax=ax[0,0], **isokwargs))
    (input.Tmean.sel(**region)
     .plot.contour(ax=ax[0,0], **TSkwargs))
    (input.ρmean.sel(**region)
     .plot.contour(ax=ax[0,0], **ρkwargs))

    (input.dSiso.sel(**region)
     .plot(ax=ax[1,0], **isokwargs))
    (input.Smean.sel(**region)
     .plot.contour(ax=ax[1,0], **TSkwargs))
    (input.ρmean.sel(**region)
     .plot.contour(ax=ax[1,0], **ρkwargs))

    (input.dTdia.sel(**region)
     .plot(ax=ax[0,1], **isokwargs))
    (input.Tmean.sel(**region)
     .plot.contour(ax=ax[0,1], **TSkwargs))
    (input.ρmean.sel(**region)
     .plot.contour(ax=ax[0,1], **ρkwargs))

    (input.dSdia.sel(**region)
     .plot(ax=ax[1,1], **isokwargs))
    (input.Smean.sel(**region)
     .plot.contour(ax=ax[1,1], **TSkwargs))
    (input.ρmean.sel(**region)
     .plot.contour(ax=ax[1,1], **ρkwargs))
    
    [aa.set_xlabel('') for aa in ax[0,:]]
    plt.tight_layout()
    
    plt.suptitle(input.attrs['dataset'], y=1.01)

    return ax


# In[13]:


region = dict(lat=-25, method='nearest')
ax = plot_gradients(annual, region)

ax[0,0].set_ylim([2000, 0])


# In[14]:


region = dict(lat=-25, method='nearest')
ax = plot_gradients(argo, region)
plt.gca().set_ylim([2000, 0])


# Looks like the ECCO product (0.5°) is a lot smoother than the Argo 1° product. Maybe the interpolation from the tiles to a regular grid has something to do with it?

# In[8]:


f, ax = plt.subplots(2, 2, sharex=True, sharey=True)

ax = ax.flat

f.set_size_inches(12, 8)

argodT.dx.sel(pres=50, method='nearest').plot(ax=ax[0], robust=True)
eccodT.dx.sel(pres=50, method='nearest').plot(ax=ax[2], robust=True)

argodT.dy.sel(pres=50, method='nearest').plot(ax=ax[1], robust=True)
eccodT.dy.sel(pres=50, method='nearest').plot(ax=ax[3], robust=True)


# ## Histograms
# 
# ECCO cannot represent the really low gradients seen in Argo. But maybe if I average out the Argo fields, I would do better? At current resolution, there is clear mesoscale signal in the fields.

# In[15]:


def log(input):
    input.values[input == 0] = np.nan
    return np.log10(input).values


tlim = [-8, -3]
slim = [-9, -4]
g = sns.jointplot(x=log(argo.dTiso),
                  y=log(annual.dTiso.interp(pres=argo.pres, 
                                            lat=argo.lat, 
                                                lon=argo.lon)),
                  kind='hex', xlim=tlim, ylim=tlim)
dcpy.plots.line45(ax=g.ax_joint)
g.ax_joint.set_xlabel('argo')
g.ax_joint.set_ylabel('ECCO')
g.fig.suptitle('Compare isopycnal |∇T| : ECCO, Argo, 2000db', y=1.05)

g = sns.jointplot(x=log(argo.dSiso),
                  y=log(annual.dSiso.interp(pres=argo.pres, 
                                            lat=argo.lat, 
                                            lon=argo.lon)),
                  kind='hex', xlim=slim, ylim=slim)
dcpy.plots.line45(ax=g.ax_joint)
g.set_axis_labels('argo', 'ECCO')
g.fig.suptitle("Compare isopycnal |∇S| : ECCO, Argo, 2000db", y=1.05)


# ## smooth argo
# 
# Smoothing the argo fields doesn't seem to make it look better?

# In[ ]:


f, ax = plt.subplots(3,1, sharex=True, sharey=True)
f.set_size_inches(8, 10)
argodS.dy.sel(pres=50, method='nearest').plot(robust=True, ax=ax[0])

smooth = argodS.dy.copy()
smooth.values = sp.ndimage.filters.gaussian_filter(smooth.values, [2,2,0])
smooth.sel(pres=50, method='nearest').plot(robust=True, ax=ax[1])

eccodT.dy.sel(pres=50, method='nearest').plot(ax=ax[2], robust=True)

