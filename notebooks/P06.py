
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-CTD-χpod-data-and-convert-to-netCDF" data-toc-modified-id="Read-CTD-χpod-data-and-convert-to-netCDF-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read CTD χpod data and convert to netCDF</a></span></li><li><span><a href="#Read-transect-+-ancillary-datasets" data-toc-modified-id="Read-transect-+-ancillary-datasets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Read transect + ancillary datasets</a></span><ul class="toc-item"><li><span><a href="#plots" data-toc-modified-id="plots-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>plots</a></span></li></ul></li><li><span><a href="#Calculate" data-toc-modified-id="Calculate-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Calculate</a></span></li></ul></div>

# # Read CTD χpod data and convert to netCDF

# In[ ]:


import sys

sys.path.append('../eddydiff/')
import eddydiff as ed


# In[131]:


etopo = xr.open_dataset('/home/deepak/datasets/ETOPO2v2g_f4.nc4', autoclose=True).drop(180., 'x')
etopo = dcpy.oceans.dataset_center_pacific(etopo.rename({'x': 'lon', 'y': 'lat'}))


# In[146]:


filename = '/home/deepak/work/eddydiff/datasets/P06/P06-means.mat'

mat = sp.io.loadmat(filename, squeeze_me=True)

p06 = xr.Dataset()

dims = ['cast', 'pres']
coords = {'cast': np.arange(300)+1, 'pres': mat['P'].astype(np.float32)}

for varname in ['chi', 'eps', 'Jq', 'KT', 'N2', 'dTdz', 'TPvar', 't', 's']:
    p06[varname] = xr.DataArray(mat[varname].T, dims=dims, coords=coords)
    
for varname in ['lat', 'lon', 'dnum', 'dist']:
    p06[varname] = xr.DataArray(mat[varname], dims=['cast'], coords={'cast':coords['cast']})

p06 = p06.rename({'t': 'T', 's': 'S', 'dnum': 'time'})
p06['time'].values = dcpy.util.datenum2datetime(p06['time'].values)
p06['rho'] = xr.DataArray(sw.dens(p06.S, p06['T'], p06['pres']), dims=dims, coords=coords)
p06 = p06.rename({'pres': 'P'})
# code expects P as pressure index vector; pres as actual 2D pressure
p06['pres'] = xr.broadcast(p06['cast'], p06['P'])[1]

p06 = p06.set_coords(['time', 'lon', 'lat', 'dist', 'pres'])

# reset co-ords to match ECCO/argo fields.
p06['lon'][p06.lon <= 0] += 360
p06 = p06.drop([1, 2], dim='cast')
p06 = p06.drop(np.arange(250, 301), dim='cast')

# fill in NaNs
p06['lon'] = p06.lon.interpolate_na(dim='cast')
p06['lat'] = p06.lat.interpolate_na(dim='cast')
p06['bathy'] = etopo.interp(lon=p06.lon, lat=p06.lat).z

p06.attrs['name'] = 'P06 merged dataset'
p06.attrs['transect_name'] = 'P06'
p06.to_netcdf('/home/deepak/work/eddydiff/datasets/P06/p06.nc')
p06


# # Read transect + ancillary datasets

# In[235]:


sys.path.append('../eddydiff/')
import eddydiff as ed

eccograd, argograd, cole = ed.read_all_datasets()

p06 = xr.open_dataset('/home/deepak/work/eddydiff/datasets/P06/p06.nc', autoclose=True)
bathy = p06.bathy.copy()
p06 = p06.where(p06['KT'] < 1e-2)
p06['bathy'] = bathy
p06['KtTz'] = p06.KT * p06.dTdz
p06


# ## plots

# In[236]:


np.log10(p06.eps).plot(x='cast')
p06.rho.plot.contour(colors='k', x='cast', yincrease=False,
                     levels=pd.qcut(p06.rho.values.ravel(), 20, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[237]:


p06['T'].plot(x='cast', cmap=mpl.cm.RdYlBu)
p06.rho.plot.contour(colors='k', x='cast', yincrease=False,
                     levels=pd.qcut(p06.rho.values.ravel(), 20, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[238]:


np.log10(p06['KT']).plot(x='cast')
p06.rho.plot.contour(colors='k', x='cast', yincrease=False,
                     levels=pd.qcut(p06.rho.values.ravel(), 20, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[198]:


dcpy.oceans.TSplot(p06.S, p06['T'], p06.pres)


# # Calculate

# In[48]:


p06rho = ed.transect_to_density_space(p06.sel(P=slice(0,2000)))


# In[242]:


eccoKe = ed.process_transect_1d(p06, eccograd, 'ECCO', nbins=18)
ed.plot_bar_Ke(eccoKe)

argoKe = ed.process_transect_1d(p06, argograd, 'ARGO', nbins=15)
ed.plot_bar_Ke(argoKe)


# In[232]:


pd.qcut(p06.rho.values.ravel(), 20, retbins=True, precision=1)


# In[188]:


cole.diffusivity.sel(**ed.get_region_from_transect(p06))


# In[178]:


clim = eccograd
transect = p06.sel(P=slice(0, 2000))

trmean, ρbins = ed.average_transect_1d(transect, nbins=20)
gradmean = ed.average_clim(clim, transect, ρbins)

Ke = ed.estimate_Ke(trmean, gradmean)
Ke


# In[179]:


def compare_transect_clim(transect, clim):

    f, ax = plt.subplots(3, 2, sharex=True, sharey=True)

    for aa, (trvar, climvar) in enumerate(zip(['T', 'S', 'rho'], 
                                             ['Tmean', 'Smean', 'ρmean'])):
        _, levels = pd.qcut(transect[trvar].values.ravel(), 20, retbins=True)
        
        (clim[climvar]
         .sel(**ed.get_region_from_transect(p06)).isel(lat=1)
         .plot.contour(ax=ax[aa, 0], levels=levels, x='lon', add_colorbar=True,
                       yincrease=False, cmap=mpl.cm.RdYlBu))
        (transect[trvar]
         .plot.contour(ax=ax[aa, 0], levels=levels, colors='k',
                                      x='lon', y='P', yincrease=False))

    trname = (transect.attrs['transect_name']
              if 'transect_name' in transect.attrs else 'transect')
    f.suptitle('Compare ' + trname + ' vs ' + clim.dataset, y=0.9)
    plt.gcf().set_size_inches((10, 12))


compare_transect_clim(p06, eccograd)
plt.gca().set_ylim((2400, 0))
#  compare_transect_clim(p06, argograd)
# plt.gca().set_ylim((2400, 0))

