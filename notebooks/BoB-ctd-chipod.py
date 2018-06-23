
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#BoB-ASIRI-2013-CTD-χpod" data-toc-modified-id="BoB-ASIRI-2013-CTD-χpod-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>BoB ASIRI 2013 CTD χpod</a></span><ul class="toc-item"><li><span><a href="#Transect-TS-plots" data-toc-modified-id="Transect-TS-plots-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Transect TS plots</a></span></li><li><span><a href="#Read-and-plot-single-transect" data-toc-modified-id="Read-and-plot-single-transect-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Read and plot single transect</a></span></li><li><span><a href="#Process-transect-data" data-toc-modified-id="Process-transect-data-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Process transect data</a></span></li></ul></li><li><span><a href="#Read-in-other-datasets" data-toc-modified-id="Read-in-other-datasets-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Read in other datasets</a></span></li><li><span><a href="#Using-climatological-gradients" data-toc-modified-id="Using-climatological-gradients-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Using climatological gradients</a></span></li><li><span><a href="#Naveira-Garabato-et-al.-(2016)-approach" data-toc-modified-id="Naveira-Garabato-et-al.-(2016)-approach-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Naveira Garabato et al. (2016) approach</a></span><ul class="toc-item"><li><span><a href="#Calculation" data-toc-modified-id="Calculation-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Calculation</a></span></li><li><span><a href="#Debugging-plots" data-toc-modified-id="Debugging-plots-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Debugging plots</a></span><ul class="toc-item"><li><span><a href="#Distributions:-dT/dz-and-dT_diapycnal" data-toc-modified-id="Distributions:-dT/dz-and-dT_diapycnal-4.2.1"><span class="toc-item-num">4.2.1&nbsp;&nbsp;</span>Distributions: dT/dz and dT_diapycnal</a></span></li></ul></li></ul></li><li><span><a href="#Ferrari-&amp;-Polzin-(2005)-approach" data-toc-modified-id="Ferrari-&amp;-Polzin-(2005)-approach-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Ferrari &amp; Polzin (2005) approach</a></span></li><li><span><a href="#Without-interpolating-to-convert-to-density-space" data-toc-modified-id="Without-interpolating-to-convert-to-density-space-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Without interpolating to convert to density space</a></span></li><li><span><a href="#Compare-various-mean-fields-/-gradients" data-toc-modified-id="Compare-various-mean-fields-/-gradients-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Compare various mean fields / gradients</a></span></li><li><span><a href="#Argo-estimate-(Cole-et-al,-2015)" data-toc-modified-id="Argo-estimate-(Cole-et-al,-2015)-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Argo estimate (Cole et al, 2015)</a></span></li><li><span><a href="#All-data-combined" data-toc-modified-id="All-data-combined-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>All data combined</a></span></li><li><span><a href="#Lessons-learned" data-toc-modified-id="Lessons-learned-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Lessons learned</a></span><ul class="toc-item"><li><span><a href="#Don't-interpolate-when-converting-to-density-space" data-toc-modified-id="Don't-interpolate-when-converting-to-density-space-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Don't interpolate when converting to density space</a></span></li><li><span><a href="#Spline-smooth-both-temperature-and-pressure" data-toc-modified-id="Spline-smooth-both-temperature-and-pressure-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Spline-smooth both temperature and pressure</a></span></li></ul></li><li><span><a href="#Groupby-on-dataframe-or-xarray-consistently" data-toc-modified-id="Groupby-on-dataframe-or-xarray-consistently-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Groupby on dataframe or xarray consistently</a></span></li><li><span><a href="#Test-gradients-in-isopycnal-planes" data-toc-modified-id="Test-gradients-in-isopycnal-planes-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Test gradients in isopycnal planes</a></span></li><li><span><a href="#Convert-transect-.mat-to-netcdf-files" data-toc-modified-id="Convert-transect-.mat-to-netcdf-files-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Convert transect .mat to netcdf files</a></span></li></ul></div>

# # BoB ASIRI 2013 CTD χpod
# 
# This notebook will attempt to infer eddy diffusivity $K_e$ from the basin-wide transect of $χ, K_T$ 
# 
# \begin{equation}
#     K_e = \frac{⟨\widetilde{χ}⟩/2 - ⟨K_T \, ∂_z \widetilde{θ}⟩ \; ∂_z θ_m}{|∇θ_m|²}
# \end{equation}

# In[2]:


import sys

sys.path.append('../eddydiff/')
import eddydiff as ed


# ## Transect TS plots

# In[57]:


tr1 = xr.open_dataset('../datasets/bob-ctd-chipod/transect_1.nc', autoclose=True)
tr2 = xr.open_dataset('../datasets/bob-ctd-chipod/transect_2.nc', autoclose=True)
tr3 = xr.open_dataset('../datasets/bob-ctd-chipod/transect_3.nc', autoclose=True)
       
_, bins = pd.cut(tr1.rho.values.ravel(), 8, retbins=True)

dcpy.oceans.TSplot(tr1.S, tr1['T'], tr1.pres, rho_levels=bins-1000)
plt.title('Transect 1')
plt.savefig('../images/bob-TS-transect-1.png', bbox_inches='tight')
dcpy.oceans.TSplot(tr2.S, tr2['T'], tr2.pres, rho_levels=bins-1000)
plt.title('Transect 2')
plt.savefig('../images/bob-TS-transect-2.png', bbox_inches='tight')
dcpy.oceans.TSplot(tr3.S, tr3['T'], tr3.pres, rho_levels=bins-1000)
plt.title('Transect 3')
plt.savefig('../images/bob-TS-transect-3.png', bbox_inches='tight')


# ## Read and plot single transect

# In[339]:


transect = (xr.open_dataset('../datasets/bob-ctd-chipod/transect_1.nc', autoclose=True)
            .sel(cast=slice(10, 36)))

transect['KtTz'] = transect['KT'] * transect['dTdz']
transect['Jq'] = 1025 * 4200 * transect['KtTz']

transect['KtTz'].values[np.abs(transect['Jq'].values) > 500] = np.nan
transect['Jq'].values[np.abs(transect['Jq'].values) > 500] = np.nan

np.log10(transect['KT']).plot(robust=True, yincrease=False)

plt.figure()
transect['dTdz'].plot(robust=True, yincrease=False)

plt.figure()
np.log10(transect['KtTz']).plot(robust=True, yincrease=False)

plt.figure()
transect['T'].sel(cast=slice(10, 36)).plot.contour(
    levels=40, colors='k', robust=True, yincrease=False)
transect['rho'].sel(cast=slice(10, 36)).plot.contour(
    levels=20, colors='r', robust=True, yincrease=False)
plt.title('black=T, red=ρ')

# transect = transect.drop(30, dim='cast').sel(P=slice(None, 110))


# Make sure $J_q$ isn't crazy

# In[340]:


np.log10(np.abs(transect['Jq'])).plot.hist()


# ## Process transect data

# In[3]:


transect = (xr.open_dataset('../datasets/bob-ctd-chipod/transect_1.nc', autoclose=True)
            .sel(cast=slice(10, 36)))

transect['KtTz'] = transect['KT'] * transect['dTdz']
transect['Jq'] = 1025 * 4200 * transect['KtTz']
transect['KtTz'].values[np.abs(transect['Jq'].values) > 1000] = np.nan
transect['Jq'].values[np.abs(transect['Jq'].values) > 1000] = np.nan

transect = transect.sel(P=slice(None, 210))

# def process_transect
# takes transect in z-space as input.

trmean_rho = ed.transect_to_density_space(transect)


# # Read in other datasets

# In[8]:


cole = ed.read_cole()

argograd = xr.open_dataset('../datasets/argo_dec_iso_gradients.nc',
                           decode_times=False, autoclose=True).load()

eccograd = xr.open_dataset('../datasets/ecco_monthly_iso_gradient.nc',
                           decode_times=False, autoclose=True).load()


# # Using climatological gradients

# \begin{equation}
#     K_e = \frac{⟨\widetilde{χ}⟩/2 - ⟨K_T \, ∂_z \widetilde{θ}⟩ \; ∂_z θ_m}{|∇θ_m|²}
# \end{equation}

# In[14]:


eccoKe = ed.process_transect_1d(transect, eccograd, 'ECCO transect1')
argoKe = ed.process_transect_1d(transect, argograd, 'ARGO transect1')

Ke = pd.DataFrame()
Ke['ecco'] = eccoKe.Ke
Ke['argo'] = argoKe.Ke
# Ke['rho'] = Ke.ecco.index.mid.astype('float')
eccoKe


# In[16]:


ed.plot_bar_Ke(eccoKe)
# ed.plot_bar_Ke(argoKe)


# In[30]:


plt.figure()
argomean.dTdia.plot(x='rho')
eccomean.dTdia.plot(x='rho')
(trmean.dTdz).plot(x='rho')
plt.legend(['argo', 'ecco', 'transect'])

plt.figure()
argomean.dTiso.plot(x='rho')
eccomean.dTiso.plot(x='rho')


# In[12]:


(ecco.dTiso.sel(lat=slice(transect.lat.min(), transect.lat.max()))
 .sel(lon=transect.lon.mean(), method='nearest')
 .plot.contourf(robust=True, yincrease=False))

(ecco.Tmean.sel(lat=slice(transect.lat.min(), transect.lat.max()))
 .sel(lon=transect.lon.mean(), method='nearest')
 .plot.contour(robust=True, colors='w', yincrease=False))

plt.gca().set_ylim([200, 0])


# # Naveira Garabato et al. (2016) approach
# 
# For turbulence quantities, the averaging operator ⟨⟩ operates *only* in the vertical direction. 
# 
# Mean temperature field is determined by spline smoothing on isopycnals and then differentiated to get "large-scale" gradients.

# ## Calculation

# In[470]:


# 1. convert to density space
#############################
trdens = ed.to_density_space(transect1)
Tdens = trdens['T']

# 2. fit cubic spline along isopycnal
# - this needs to take ρbins as input?
#####################################
Tdens_i = ed.smooth_cubic_spline(trdens['T'], False)
Pdens_i = ed.smooth_cubic_spline(trdens['P'], False)
Tsmooth = ed.to_depth_space(Tdens_i, Pold=Pdens_i, Pnew=None)

# 3. calculate gradients with smoothed field
############################################
dTiso, dTdia = ed.calc_iso_dia_gradients(Tdens_i, Pdens_i)
dTmdz = ed.to_density_space(-ed.xgradient(Tsmooth, 'P'))
dTmdz.name = 'dz'

# 4. Bin smoothed gradients by density
######################################
ρbins = Tdens_i.rho[::2]
dTdf = (ed.exchange(xr.merge([dTdia, dTiso]),
                    {'cast': 'dist'})
        .to_dataframe()
        .reset_index())
dTmean = ed.bin_avg_in_density(dTdf, ρbins)
transmean = ed.bin_avg_in_density(trdens.to_dataframe().reset_index(), ρbins)

# 5. do estimate
################
transKe = xr.Dataset()
transKe['Ke'] = (transmean.chi/2 - transmean.KtTz * dTmean.dia)/dTmean.iso**2
transKe.Ke.name = '$K_e$'
transKe['KT'] = transmean.KT
transKe['KtTz'] = transmean.KtTz
transKe['dTdz'] = transmean.dTdz
transKe['chi'] = transmean.chi
transKe['dTmdz'] = dTmean.dia
transKe['dTiso'] = dTmean.iso
transKe['rho'].values = np.round(transKe.rho.values, decimals=2)
transKe.attrs.name = 'BoB CTD χpod'

ed.plot_transect_Ke(transKe)

ed.plot_bar_Ke(transKe.isel(cast=20).to_dataframe())


# ## Debugging plots

# In[471]:


trdens = ed.to_density_space(transect1)
Tdens = trdens['T']trdens['P'].plot.contour(levels=20)
dcpy.ts.xfilter(trdens['P'], dim='cast', flen=10).plot.contour(
    colors='r', levels=20)
Pdens_i.plot.contour(levels=20, colors='k')


# ### Distributions: dT/dz and dT_diapycnal

# The distributions of (dT/dz)_ρ is different from (dT/dz)_z which is maybe not surprising but the difference is a factor of 2, this makes me suspicious.

# In[73]:


dTi, dTd = ed.calc_iso_dia_gradients(trdens['T'], trdens['P'])
dTi_i, dTd_i = ed.calc_iso_dia_gradients(Tdens_i, Pdens_i)

histargs = {'histtype': 'step'}

f, ax = plt.subplots(1, 2)
dTi.plot.hist(ax=ax[0], **histargs)
dTi_i.plot.hist(ax=ax[0], **histargs)
ax[0].legend(['dTiso of T interp to ρ space',
              'dTiso of spline smooth T'])

dTd.plot.hist(ax=ax[1], **histargs)
dTd_i.plot.hist(ax=ax[1], **histargs)
trdens['dTdz'].plot.hist(ax=ax[1], **histargs)
transect['dTdz'].plot.hist(ax=ax[1], **histargs)
dTmdz.plot.hist(ax=ax[1], **histargs)
plt.gca().legend(['dTdia of T interp to ρ space',
                  'dTdia of spline smooth T',
                  'local dTdz interp to ρ space',
                  'observed local dTdz',
                  'dTdz of spline smooth T after moving to z space'])

f.set_size_inches(9, 5)


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(xr.broadcast(trdens['dist'], Pdens_i)[0].T,
            Pdens_i, 20, dTd_i, cmap=mpl.cm.RdYlBu)
plt.gca().invert_yaxis()
plt.colorbar()


transect['T'].plot.contour(colors='r', levels=30, yincrease=False)
transect['rho'].plot.contour(levels=25, colors='k', yincrease=False)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(xr.broadcast(trdens['dist'], Pdens_i)[0].T,
            Pdens_i, 20, dTd_i, cmap=mpl.cm.RdYlBu)
plt.gca().invert_yaxis()
plt.colorbar()


# In[326]:


###################################
# sample mean fields along transect
###################################
seltrans = dict(lon=transect.lon, lat=transect.lat, method='nearest')

f, ax = plt.subplots(4, 1)
f.set_size_inches(8, 14)

(eccograd.dTiso.sel(pres=slice(200, 0, -1)).sel(**seltrans)
 .plot(ax=ax[0], cmap=mpl.cm.Reds))
(eccograd.Tmean.sel(pres=slice(200, 0, -1)).sel(**seltrans)
 .plot.contour(ax=ax[0], levels=20, colors='w'))
ax[0].set_ylim([200, 0])

# plots
Tdens.plot.contourf(levels=50, ax=ax[1], cmap=mpl.cm.RdYlBu_r, yincrease=False)
Tdens_i.plot.contour(levels=50, colors='k', ax=ax[1], yincrease=False)
trdens.P.plot.contour(levels=20, colors='lightgray', ax=ax[1], yincrease=False)


np.abs(dTiso).plot(ax=ax[2], cmap=mpl.cm.Reds, robust=True, yincrease=False)
Tdens_i.plot.contour(levels=50, colors='k', ax=ax[2], yincrease=False)
#trdens.P.plot.contour(levels=20, colors='lightgray', ax=ax[2], yincrease=False)

dTdia.plot(ax=ax[3], robust=True, yincrease=False)
Tdens_i.plot.contour(levels=50, colors='k', ax=ax[3], yincrease=False)
Pdens_i.plot.contour(levels=20, colors='lightgray', ax=ax[3], yincrease=False)


# In[18]:


f, ax = plt.subplots(2,1)
f.set_size_inches(8,6)
trdens['P'].plot.contour(ax=ax[0], yincrease=False, levels=50, colors='k')
Pdens_i.plot.contour(ax=ax[0], yincrease=False, levels=40, cmap=mpl.cm.Blues)
transect['rho'].plot.contour(ax=ax[1], yincrease=False, levels=50)
ax[1].set_ylim([100, 0])


# # Ferrari & Polzin (2005) approach
# 
# They average all data on density surfaces, estimate mean depth of isopycnal $z_n$ and calculate diapycnal gradient. The result is a vertical profile of eddy diffusivity. All spatial variations are averaged over.
# 
# They calculate an isopycnal gradient by doing a plane fit along isopycnals to all the data. This is required for the finestructure estimate

# In[76]:


transKe


# In[323]:


# def transect_fp(trdens):

trdens = ed.to_density_space(transect)

dTiso = xr.DataArray(np.ones_like(trdens.rho) * np.nan,
                     dims=['rho'], coords={'rho': trdens.rho})

# fit straight lines to get mean gradients 
# FP do a plane fit in 2D
for idx, rr in enumerate(trdens.rho):
    Tvec = trdens['T'].sel(rho=rr)
    mask = ~np.isnan(Tvec)
    
    if mask.sum() < 3:
        continue
        
    dTiso[idx], _, _, _, _ = sp.stats.linregress(trdens.dist[mask]*1000, Tvec[mask] )

trmean = trdens.mean(dim='cast')
trmean

Tmean = ed.to_depth_space(trmean['T'].expand_dims(['cast']), 
                          Pold=trmean['P'].expand_dims(['cast'])).mean(dim='cast')
rhomean = ed.to_depth_space(trmean['rho'].expand_dims(['cast']), 
                            Pold=trmean['P'].expand_dims(['cast'])).mean(dim='cast')

_, Tsmooth = ed.fit_spline(Tmean.P, Tmean, k=2)

# Tmean.plot()
dTdz = ed.xgradient(-Tsmooth, dim='P')
dTdz.coords['rho'] = rhomean

dTdz = dTdz.expand_dims(['cast'])
dTdz['rho'] = dTdz.rho.expand_dims(['cast'])

dTmdz = ed.to_density_space(dTdz).mean(dim='cast')

fpKe = xr.Dataset()
fpKe['chi'] = trmean.chi
fpKe['dTmdz'] = trmean.dTdz
fpKe['dTiso'] = dTiso
fpKe['KtTz'] = trmean.KtTz
fpKe['KT'] = trmean.KT
fpKe['dTdz'] = trmean.dTdz

fpKe['Ke'] = np.abs(fpKe.chi/2 - fpKe.KtTz * fpKe.dTmdz)/fpKe.dTiso**2

ed.plot_bar_Ke(fpKe.to_dataframe())


# # Without interpolating to convert to density space

# In[472]:


mask= ~np.isnan(trmean_rho['T'].isel(rho=5))
ed.fit_spline(trmean_rho['dist'][mask].values, 
              trmean_rho['T'].isel(rho=5)[mask].values, debug=True)


# In[449]:


# Run Section 1.3 first
# 2. Generate mean field T, P using cubic splines to smooth
#    Calculate Trms as RMS of deviations from smoothed T field

trmean_rho['Tmean'] = ed.smooth_cubic_spline(trmean_rho['T'])
trmean_rho.Tmean.attrs['name'] = 'Smoothed temperature'

trmean_rho['Pmean'] = ed.smooth_cubic_spline(trmean_rho['P'])
trmean_rho.Pmean.attrs['name'] = 'Smoothed pressure'


def check_spline_smoothing(T, Ts, P, Ps, Traw, Praw):
    f, ax = plt.subplots(3, 1)

    T.plot.contour(ax=ax[0], y='rho', levels=50, colors='r')
    Ts.plot.contour(ax=ax[0], y='rho', levels=50, colors='k', yincrease=False)

    P.plot.contour(ax=ax[1], y='rho', levels=50, colors='r')
    Ps.plot.contour(ax=ax[1], y='rho', levels=50, colors='k', yincrease=False)

    #cmat, rhomat = xr.broadcast(T.cast, T.rho)
    #ax[2].contour(cmat, P, T, levels=50, colors='r')

    Tz = ed.to_depth_space(T, P)
    Tzs = ed.to_depth_space(Ts, Ps)

    ax[2].contour(xr.broadcast(Traw.cast, Praw)[0], Praw.T, Traw.T, 50,
                  cmap=mpl.cm.BuGn)
    Tz.plot.contour(ax=ax[2], x='cast', levels=30, colors='r')
    Tzs.plot.contour(ax=ax[2], x='cast', levels=30,
                     colors='k', yincrease=False)

    f.set_size_inches((8, 10))
    ax[0].set_title('Smoothed temp')
    ax[1].set_title('Smoothed pressure')
    ax[2].set_title('Raw and smoothed temperature in depth space')
    ax[2].set_ylim([220, 0])
    plt.tight_layout()


check_spline_smoothing(trmean_rho['T'], trmean_rho['Tmean'],
                       trmean_rho['P'], trmean_rho['Pmean'],
                       transect['T'], transect['pres'])

trmean_rho['Trms'] = np.sqrt(
    ((trmean_rho['T'] - trmean_rho['Tmean'])**2).mean(dim='cast'))
trmean_rho.Trms.attrs['name'] = 'RMS temp variations'


# In[515]:


# 3. Calculate isopyncal and diapycnal gradients of the mean field
trmean_rho['dTiso'], trmean_rho['dTdia'] =     ed.calc_iso_dia_gradients(trmean_rho['Tmean'], trmean_rho['Pmean'], debug=True)
trmean_rho.dTiso.attrs['name'] = 'Isopycnal ∇T'
trmean_rho.dTdia.attrs['name'] = 'Diapycnal ∇T'


# In[516]:


transKe = xr.Dataset()
transKe['KT'] = trmean_rho.KT
transKe['KtTz'] = trmean_rho.KtTz
transKe['dTdz'] = trmean_rho.dTdz
transKe['chi'] = trmean_rho.chi
transKe['dTmdz'] = trmean_rho.dTdia
transKe['dTiso'] = trmean_rho.dTiso
transKe['rho'].values = np.round(transKe.rho.values, decimals=2)
transKe['Tm'] = trmean_rho['Tmean']
transKe.attrs['name'] = 'BoB CTD χpod'


def calc_Ke(transKe, navg=None):

    transKe = transKe.copy()

    if navg is not None and np.isinf(navg):
        transKe = transKe.mean(dim='cast')

    elif navg is not None:
        cbins = np.arange(transKe.cast.min(), transKe.cast.max()+1, navg)
        transKe = (transKe.groupby_bins('cast', cbins, labels=cbins[:-1]+1)
                   .mean(dim='cast'))

     
    transKe['Ke'] = (transKe.chi/2 - transKe.KtTz *
                     transKe.dTmdz)/transKe.dTiso**2
    transKe.Ke.values[np.abs(transKe.dTiso.values) < 5e-7] = np.nan
    transKe.Ke['name'] = '$K_e$'
    
    transKe.attrs['navg'] = navg
    
    if 'cast_bins' in transKe.coords:
        transKe = transKe.rename({'cast_bins': 'cast'})

    return transKe


transKe = calc_Ke(transKe)

ed.plot_transect_Ke(transKe)


# In[452]:



transKe2 = calc_Ke(transKe, navg=None)
ax, axback = ed.plot_transect_var(x='cast', y='rho', data=transKe2.Ke,
                               fill=trmean_rho['dTiso'], contour=trmean_rho['P'],
                               xlim=[1, 1e4], xticks=[1e3])
ax, axback = ed.plot_transect_var(x='cast', y='rho', data=transKe2.chi/2, 
                               bar2=transKe2.KtTz * transKe2.dTmdz,
                               fill=trmean_rho['T'], contour=trmean_rho['P'])


# In[439]:


transKe2.Ke.plot.line(y='rho', hue='cast', yincrease=False)
plt.gca().set_xscale('log')
plt.gca().set_xlim([1e2, 1e5])

cole = ed.read_cole()

region = ed.get_region_from_transect(transect)

cole_bay = (cole.sel(lat=region['lat'])).mean(dim='lon').mean(dim='lat')
cole_bay['density_mean_depth'] += 1000
cole_bay = cole_bay.set_coords('density_mean_depth')
cole_bay.diffusivity.plot(y='density_mean_depth', color='k', yincrease=False)


# In[517]:


transKe2.Ke.where(transKe2.Ke > 0).mean(dim='cast').plot.line(y='rho')
cole_bay = (cole.sel(lat=region['lat'])).mean(dim='lon').mean(dim='lat')
cole_bay['density_mean_depth'] += 1000
cole_bay = cole_bay.set_coords('density_mean_depth')
cole_bay.diffusivity.plot(y='density_mean_depth', color='k', yincrease=False)

plt.gca().set_xscale('log')
plt.gca().set_xlim([1e2, 1e5])
plt.figlegend(['Mean in along-transect direction', 'Cole et al (2015)'])


# # Compare various mean fields / gradients

# In[34]:


argo = xr.open_dataset('../datasets/argo_clim_iso_gradients.nc',
                       decode_times=False, autoclose=True).load()

ecco = xr.open_dataset('../datasets/ecco_annual_iso_gradient.nc',
                       autoclose=True).load()


# In[44]:


region = ed.get_region_from_transect(transect)

argo.dTiso.sel(**region).mean(dim='lon').plot(robust=True, yincrease=False)
plt.figure()
ecco.dTiso.sel(**region).mean(dim='lon').plot(robust=True, yincrease=False)

plt.figure()
np.abs(trmean_rho.dTiso).plot(robust=True, yincrease=False)


# # Argo estimate (Cole et al, 2015)

# In[459]:


region = ed.get_region_from_transect(transect)

cole_bay = (cole.sel(lat=region['lat'], lon=region['lon']))

(cole_bay.diffusivity.mean(dim='lon')
 .plot(y='depth', yincrease=False,
       norm=mpl.colors.LogNorm(),
       cmap=mpl.cm.Reds))
plt.gca().set_ylim([200, 0])


# In[513]:


(cole_bay.diffusivity.mean(dim='lat').mean(dim='lon')
 .plot.line(y='depth', yincrease=False))
plt.gca().set_xscale('log')
plt.gca().set_xlim([1e2, 3e4])
plt.gca().set_ylim([200, 0])
plt.gca().grid(True)


# In[514]:


(cole_bay.diffusivity.groupby_bins(cole_bay.density_mean_depth+1000,
                                   ρbins,
                                   labels=(ρbins[:-1]+ρbins[1:])/2).mean()
 .plot.line(y='density_mean_depth_bins', yincrease=False))


# # All data combined

# Let's ignore 3 for now, not sure why that overlaps.

# In[6]:


transect1 = (xr.open_dataset('../datasets/bob-ctd-chipod/transect_1.nc', autoclose=True))

transect2 = (xr.open_dataset('../datasets/bob-ctd-chipod/transect_2.nc', autoclose=True))

transect3 = (xr.open_dataset('../datasets/bob-ctd-chipod/transect_3.nc', autoclose=True))

plt.plot(transect1.lon, transect1.lat, '.', ms=20)
plt.plot(transect2.lon, transect2.lat, '.', ms=20)
plt.plot(transect3.lon, transect3.lat, 'o')
plt.legend(('big(1)', 'big(2)', 'big(3)'))


# In[9]:


# merge transect 1 and transect2
transect = pd.concat([tr.to_dataframe().reset_index()
                      for tr in [transect1.sel(P=slice(0, 120))]])

transect['KtTz'] = transect['KT'] * transect['dTdz']
transect['Jq'] = 1025 * 4200 * transect['KtTz']

mask = np.logical_or(np.abs(transect['Jq'].values) > 2000,
                     transect['KT'].values > 5e-3)

transect['KtTz'].values[mask] = np.nan
transect['Jq'].values[mask] = np.nan
transect['chi'].values[mask] = np.nan

eccoKe2 = ed.process_transect_1d(transect, eccograd, 'ECCO')
argoKe2 = ed.process_transect_1d(transect, argograd, 'ARGO')

ed.plot_bar_Ke(eccoKe2)
ed.plot_bar_Ke(argoKe2)

eccoKe2.Ke


# In[467]:


transect1['T'].sel(cast=slice(9, 40)).plot(cmap=mpl.cm.RdYlBu_r, yincrease=False)
transect1['rho'].sel(cast=slice(9, 40)).plot.contour(colors='k', 
                                                     levels=np.arange(1020, 1028), 
                                                     yincrease=False)
plt.figure()

(eccograd['ρmean'].sel(**ed.get_region_from_transect(transect1))
 .mean(dim='lon')
 .plot.contour(levels=np.arange(1020, 1028), colors='k', yincrease=False))
plt.gca().set_ylim([200, 0])


# In[468]:


trmean, ρbins = ed.average_transect_1d(transect)
    
eccomean = ed.average_clim(eccograd, transect, ρbins)

eccomean


# In[22]:


sp, Tsm = ed.fit_spline(eccomean['Pmean'], eccomean['Tmean'], k=3, debug=False)
plt.plot(sp.derivative(1)(eccomean['Pmean']))
plt.plot(-eccomean.dTdz.values)


# # Lessons learned

# ## Don't interpolate when converting to density space
# 
# +The fields and gradients are different!+
# 
# *FALSE ALARM: This is because transect.P and transect.pres are different!* 

# In[39]:


# transform to depth space, calculate gradient and transform back
Tsmooth = ed.to_depth_space(Tdens_i, Pold=Pdens_i, Pnew=None)
dTmdz = (-ed.xgradient(Tsmooth, 'P'))
ed.to_density_space(dTmdz).plot(yincrease=False)

# original calculation ΔT and ΔP are estimated in density space
plt.figure()
trmean_rho.dTdia.plot(yincrease=False, x='cast')


# ## Spline-smooth both temperature and pressure
# 
# Below I calculate dT/dP in density space and compare that to dT/dP calculated by 
# 1. transforming spline smoothed T to pressure space
# 2. differentiate to get dT/dP
# 3. convert back to density space and plot.

# In[38]:


trdens = ed.to_density_space(transect)

plt.close('all')

Tdens_i = ed.smooth_cubic_spline(trdens['T'], False)
Pdens_i = ed.smooth_cubic_spline(trdens['P'], False)

levels = np.linspace(trdens['T'].min(), trdens['T'].max(), 40)
trdens['T'].plot.contour(colors='r', levels=levels)
Tdens_i.plot.contour(colors='k', levels=levels, yincrease=False)
plt.title('In ρ space')

plt.figure()
Tsmooth = ed.to_depth_space(Tdens_i, Pold=Pdens_i, Pnew=None)
transect['T'].plot.contour(colors='r', levels=levels, yincrease=False)
Tsmooth.plot.contour(colors='k', levels=levels, yincrease=False)
plt.title('To depth space using spline smoothed pressure')
plt.ylim([120, 0])

plt.figure()
psmooth = dcpy.ts.xfilter(trdens['P'], dim='cast', flen=10)
Tsmooth = ed.to_depth_space(Tdens_i, Pold=psmooth, Pnew=None)
transect['T'].plot.contour(colors='r', levels=levels, yincrease=False)
Tsmooth.plot.contour(colors='k', levels=levels, yincrease=False)
plt.title('To depth space using hann smoothed pressure')
plt.ylim([120, 0])


plt.figure()
Tsmooth = ed.to_depth_space(Tdens_i, Pold=trdens['P'], Pnew=None)
transect['T'].plot.contour(colors='r', levels=levels, yincrease=False)
Tsmooth.plot.contour(colors='k', levels=levels, yincrease=False)
plt.title('To depth space using unsmoothed pressure')
plt.ylim([120, 0])


# # Groupby on dataframe or xarray consistently
# 
# xarray seems to use a different kind of index with groupby_bins.

# # Test gradients in isopycnal planes

# In[25]:


coords = {'cast': np.linspace(1, 50, 50),
          'P': np.linspace(0, 500, 200)}

T = xr.DataArray(np.ones((len(coords['P']), len(coords['cast']))) * np.nan,
                 dims=['P', 'cast'], coords=coords, name='T')

T = - 0.6 * T.P + 0.05 * T.cast
T.name = 'T'

rho = 1025 * (1 - 1.7e-4 * (T-15))
rho.name = '$ρ$'

T.plot.contourf(levels=20)
rho.plot.contour(colors='k', yincrease=False)

dT = ed.gradient(T.rename({'P': 'z', 'cast': 'x'}))
dT['dy'] = xr.zeros_like(dT['dx'])
dT.attrs['name'] = 'dT'

drho = ed.gradient(rho.rename({'P': 'z', 'cast': 'x'}))
drho['dy'] = xr.zeros_like(drho['dx'])
drho.attrs['name'] = 'dρ'

dT


# # Convert transect .mat to netcdf files

# In[5]:


ed.convert_mat_to_netcdf()

