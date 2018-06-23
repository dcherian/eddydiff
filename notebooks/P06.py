
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Read-CTD-χpod-data-and-convert-to-netCDF" data-toc-modified-id="Read-CTD-χpod-data-and-convert-to-netCDF-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Read CTD χpod data and convert to netCDF</a></span></li><li><span><a href="#Read-transect-+-ancillary-datasets" data-toc-modified-id="Read-transect-+-ancillary-datasets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Read transect + ancillary datasets</a></span><ul class="toc-item"><li><span><a href="#plots" data-toc-modified-id="plots-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>plots</a></span></li></ul></li><li><span><a href="#Calculate" data-toc-modified-id="Calculate-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Calculate</a></span></li><li><span><a href="#Calculate-2" data-toc-modified-id="Calculate-2-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Calculate 2</a></span></li></ul></div>

# # Read CTD χpod data and convert to netCDF

# In[94]:


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
p06['sigma_0'] = xr.DataArray(sw.pden(p06.S, p06['T'], p06['pres'], 0), dims=dims, coords=coords)
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

# add in bathymetry
etopo = xr.open_dataset('/home/deepak/datasets/ETOPO2v2g_f4.nc4', autoclose=True).drop(180., 'x')
etopo = dcpy.oceans.dataset_center_pacific(etopo.rename({'x': 'lon', 'y': 'lat'}))
p06['bathy'] = etopo.interp(lon=p06.lon, lat=p06.lat).z

p06.attrs['name'] = 'P06 merged dataset'
p06.attrs['transect_name'] = 'P06'
p06.to_netcdf('/home/deepak/work/eddydiff/datasets/P06/p06.nc')
p06


# # Read transect + ancillary datasets

# In[288]:


sys.path.append('../eddydiff/')
import eddydiff as ed

p06 = xr.open_dataset('/home/deepak/work/eddydiff/datasets/P06/p06.nc', autoclose=True)
bathy = p06.bathy.copy()
p06 = p06.where(p06['KT'] < 5e-3)
p06['bathy'] = bathy
p06['KtTz'] = p06.KT * p06.dTdz
p06

eccograd, argograd, cole = ed.read_all_datasets(kind='monthly', transect=p06)


# In[2]:


argograd.Smean.plot(x='cast', yincrease=False)


# ## plots

# In[259]:


np.log10(p06.eps).plot(x='cast')
p06.rho.plot.contour(colors='k', x='cast', yincrease=False,
                     levels=pd.qcut(p06.rho.values.ravel(), 20, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[6]:


p06['T'].plot(x='cast', cmap=mpl.cm.RdYlBu)
p06.rho.plot.contour(colors='k', x='cast', yincrease=False,
                     levels=pd.qcut(p06.rho.values.ravel(), 20, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[284]:


np.log10(p06['KT']).plot(x='cast')
p06.rho.plot.contour(colors='k', x='cast', yincrease=False, labels=True,
                     levels=pd.qcut(p06.rho.values.ravel(), 15, retbins=True)[1])
plt.gca().fill_between(p06.cast, 6100, -p06.bathy, color='k', zorder=10)
plt.gcf().set_size_inches(10, 10/1.6)


# In[355]:


bins = [1024.6, 
        1025.6, 1025.8, 
        1026.0, 1026.25, 1026.5, 1026.75, 
        1027.0, 1027.1, 1027.2, 1027.3, 1027.4, 1027.5, 1027.6, 1027.7, 1027.8, 1027.9]
dcpy.oceans.TSplot(p06.S, p06['T'], p06.pres, rho_levels=bins)


# In[61]:


np.log10(p06.Jq).plot.hist(bins=30)


# # Calculate

# In[283]:


nbins = 18

p06['rho'] = p06['sigma_0']
eccograd['ρmean'] = eccograd['sigma_0']


eccoKe = ed.process_transect_1d(p06, eccograd, 'ECCO', nbins=nbins)
ed.plot_bar_Ke(eccoKe, Ke_log=True)

argoKe = ed.process_transect_1d(p06, argograd, 'ARGO', nbins=nbins)
ed.plot_bar_Ke(argoKe, Ke_log=True)


# # Calculate 2

# In[383]:


sys.path.append('../eddydiff/')
import eddydiff as ed

p06 = xr.open_dataset(
    '/home/deepak/work/eddydiff/datasets/P06/p06.nc', autoclose=True)
bathy = p06.bathy.copy()
p06 = p06.where(p06['KT'] < 1e-3)
p06['bathy'] = bathy
p06['KtTz'] = p06.KT * p06.dTdz
p06

if 'eccograd' not in locals():
    eccograd, argograd, cole = ed.read_all_datasets(
        kind='monthly', transect=p06)

# use sigma_0
p06['rho'] = p06['sigma_0']
eccograd['ρmean'] = eccograd['sigma_0']

bins = [1024.6,
        1025.6, 1025.8,
        1026.0, 1026.15, 1026.3, 1026.45, 1026.6, 1026.8, 
        1027.0, 1027.1, 1027.2, 1027.3, 1027.4, 1027.5, 1027.6, 1027.7, 1027.8, 1027.9]
trdens, bins = ed.bin_to_density_space(p06, bins=bins)

eccograd['dist'] = p06.dist
eccodens, _ = ed.bin_to_density_space(eccograd.rename({'ρmean': 'rho'}), bins)

newKe = xr.Dataset()
newKe['KT'] = trdens.KT.mean(dim='cast')
newKe['chi'] = trdens.chi.mean(dim='cast')
newKe['KtTz'] = trdens.KtTz.mean(dim='cast')
newKe['dTiso'] = np.abs(eccodens.dTiso.mean(dim='cast'))
newKe['dTmdz'] = np.abs(eccodens.dTdz.mean(dim='cast'))
newKe['dTdz'] = trdens.dTdz.mean(dim='cast')

newKe['Ke'] = ((newKe['chi']/2 - np.abs(newKe['KtTz'] * newKe['dTmdz']))
               / (newKe['dTiso']**2))

ed.plot_bar_Ke(newKe.to_dataframe())


# The above really awesome plot results after a few tweaks.
# 
# 1. Doing things in σ_0.
# 
# 2. Handcrafted density bins using the T-S diagram as guide.
#    This is really important. Had to fiddle around slightly to not get negative
#    values. Biggest improvement is that I can now resolve thermocline and coarsen
#    the abyss. This is necessary because the big signal is in the thermocline and
#    the abyssal values need to account for topography. Automatically choosing
#    bins by pandas.cut/pandas.qcut would put too many bins down deep and fewer in
#    the thermocline.
# 
# 3. Throwing out all KT > 1e-3. This is really important. Will need to QC the
#    values coming out of the CTD_chipod analysis code.
# 
# 4. Redid (vectorized) the bin-averaging by cast. This is a better way to do it I
#    think. Have to use pandas so I can groupby using multiple variables.
# 
# 5. I tried fitting straight lines to the bin-averaged ECCO field to get dTiso
#    but this seems to under-estimate values. Currently I bin-average dTiso and
#    use that. This might not be crazy because it mirrors what I do with χ and we
#    know that χ has to go along with the appropriate local gradients. i.e. if
#    high χ coincides with high dTiso locally, we want the averaged dTiso to be
#    biased high since averaged χ will be biased high.

# In[393]:


f, ax = plt.subplots(2, 2, constrained_layout=True)

Trms = trdens['T'].std(dim='cast')
Trms.name = 'RMS potential temp.'
Trms.attrs['units'] = '°C'
Trms.plot(ax=ax[0, 0], y='rho', yincrease=False)
newKe.Ke.where(newKe.Ke > 0).plot(
    marker='.', ax=ax[0, 1], y='rho', yincrease=False)
ax[0, 0].set_ylabel('$σ_0$ [kg/m$^3$]')
ax[0, 1].set_ylabel('$σ_0$ [kg/m$^3$]')
ax[0, 1].set_xlabel('$K_e [m^2/s^2]$ ')

ax[1, 0].plot(newKe.Ke.where(newKe.Ke > 0), trdens.P.mean(dim='cast'), '.-')
ax[1, 0].plot(cole.diffusivity.mean(dim='cast'), cole.P)
ax[1, 0].legend(['χpod estimate', 'Cole et al. (2015)'])
ax[1, 0].invert_yaxis()
ax[1, 0].set_ylabel('Pressure [dbar]')
ax[1, 0].set_xlabel('$K_e [m^2/s^2]$ ')

dcpy.oceans.TSplot(p06.S, p06['T'], p06.pres,
                   rho_levels=bins, ax=ax[1, 1], ms=1)

f.set_size_inches(8, 10)


# Linear fits to ECCO T along ρ surfaces in the along-transect direction doesn't work. The field has too much curvature

# In[300]:


# recalculate gradients from mean fields in density space
dTiso = xr.ones_like(eccodens.rho) * np.nan
dTiso.name = 'dTiso'
dTdz = xr.ones_like(eccodens.rho) * np.nan
dTdz.name = 'dTdz'

for idx, rr in enumerate(eccodens.rho):
    Tvec = eccodens.Tmean.sel(rho=rr)
    mask = np.isnan(Tvec)
    if len(Tvec[~mask]) > 0:
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(eccodens.dist[~mask]*1000,
                                                                          Tvec[~mask])
    dTiso[idx] = slope

dTdz.values = -np.gradient(eccodens.Tmean.mean(dim='cast'),
                           eccodens.P.mean(dim='cast'))

f, ax = plt.subplots(1, 2)
np.abs(dTiso).plot.line(ax=ax[0], y='rho')
np.abs(eccodens.dTiso).mean(dim='cast').plot.line(
    ax=ax[0], y='rho', yincrease=False)
ax[0].set_xlim([0, 5e-6])
f.legend(['recalculated from density-averaged field',
          'along-transect mean of local gradient'])

dTdz.plot(ax=ax[1], y='rho', yincrease=False)
eccodens.dTdz.mean(dim='cast').plot.line(ax=ax[1], y='rho', yincrease=False)


# In[290]:


nn = 6

Tvec = eccodens.Tmean.isel(rho=nn)
mask = np.isnan(Tvec)
plt.plot(eccodens.dist[~mask]*1000, Tvec[~mask], '.')

plt.plot(eccodens.dist*1000, Tvec.mean() + dTiso.isel(rho=nn) *  (eccodens.dist-eccodens.dist.mean()) * 1000)


# In[56]:


eccograd.dTiso.plot(x='cast', yincrease=False, robust=True)


# In[22]:


clim = eccograd.copy()

clim = clim.to_dataframe().reset_index()

trmean, ρbins = ed.average_transect_1d(p06, nbins=15)
    
climrho = clim.groupby(pd.cut(clim.ρmean, ρbins, precision=1))


# In[57]:


np.log10(cole.diffusivity).plot(x='cast', yincrease=False)


# In[60]:


cole.diffusivity.mean(dim='cast').plot()


# In[40]:


clim = eccograd.sel(time=9)
transect = p06.sel(P=slice(0, 2000))

trmean, ρbins = ed.average_transect_1d(transect, nbins=20)
gradmean = ed.average_clim(clim, transect, ρbins)

Ke = ed.estimate_Ke(trmean, gradmean)
Ke


# In[19]:


def compare_transect_clim(transect, clim):

    f, ax = plt.subplots(3, 1, sharex=True, sharey=True)

    for aa, (trvar, climvar) in enumerate(zip(['T', 'S', 'rho'],
                                              ['Tmean', 'Smean', 'ρmean'])):
        _, levels = pd.qcut(transect[trvar].values.ravel(), 20, retbins=True)

        (clim[climvar]
         .sel(**ed.get_region_from_transect(p06)).isel(lat=1)
         .plot.contour(ax=ax[aa], levels=levels, x='lon', add_colorbar=True,
                       yincrease=False, cmap=mpl.cm.RdYlBu))
        (transect[trvar]
         .plot.contour(ax=ax[aa], levels=levels, colors='k',
                       x='lon', y='P', yincrease=False))

    trname = (transect.attrs['transect_name']
              if 'transect_name' in transect.attrs else 'transect')
    f.suptitle('Compare ' + trname + ' vs ' + clim.dataset, y=0.9)
    plt.gcf().set_size_inches((10, 12))


compare_transect_clim(p06, eccograd.sel(time=8))
plt.gca().set_ylim((2400, 0))
#  compare_transect_clim(p06, argograd)
# plt.gca().set_ylim((2400, 0))

