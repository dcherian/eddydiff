import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr


def exchange(input, kwargs):

    for kk in kwargs.keys():
        d1 = input[kk]
        d2 = input[kwargs[kk]]

        output = (input.copy().drop([d1.name, d2.name])
                  .rename({d1.name: d2.name}))

    output.coords[d2.name] = d2.values
    output.coords[d1.name] = xr.DataArray(
        d1.values, dims=[d2.name], coords={d2.name: d2.values})

    return output


def format_ecco(input):

    output = (input.rename({'dep': 'z'})
              .drop(['lon', 'lat'])
              .rename({'i1': 'time', 'i2': 'pres'}))
    output['pres'] = output['z']
    output['i4'] = input['lon'].values[0, :].squeeze()
    output['i3'] = input['lat'].values[:, 0].squeeze()

    output = (output
              .rename({'i4': 'lon', 'i3': 'lat'})
              .drop(['z'])
              .set_coords(['lon', 'lat']))

    return output


def gradient(input):
    ''' Given an input DataArray, calculate gradients in three directions
        and return it.

        Output
        ======

        dx, dy, dz = gradients in x,y,z
    '''

    gradients = np.gradient(input, *[input.coords[dim] for dim in input.dims])

    grads = xr.Dataset()
    for idx, dim in enumerate(input.dims):
        grads['d'+dim] = xr.DataArray(gradients[idx], dims=input.dims,
                                      coords=input.coords)

    grads['mag'] = xr.zeros_like(grads.data_vars['d'+dim])
    for var in grads.data_vars:
        if var == 'mag':
            continue

        grads['mag'] += grads[var]**2

    grads['mag'] = np.sqrt(grads['mag'])

    return grads


def wrap_gradient(input):
    ''' Given an input DataArray, calculate gradients in three directions
        and return it.

        Output
        ======

        dx, dy, dz = gradients in x,y,z
    '''

    gradients = np.gradient(input, input['pres'], input['lat'], input['lon'])
    dims = ['pres', 'lat', 'lon']

    coords = input.coords
    grads = xr.Dataset()
    grads['dz'] = xr.DataArray(gradients[0], dims=dims, coords=coords)*-1
    grads['dy'] = xr.DataArray(gradients[1], dims=dims, coords=coords)/1e5
    grads['dx'] = xr.DataArray(gradients[2], dims=dims, coords=coords)/1e5

    grads['mag'] = np.sqrt(grads.dx**2 + grads.dy**2 + grads.dz**2)

    return grads


def project_vector(vector, proj, kind=None):
    ''' Project 'vector' along 'proj'.

        Returns
        =======

        List of DataArrays (vector projection, rejection)
    '''

    def dot_product(a, b):
        ''' dot product of 2 gradient vectors '''
        return a.dx * b.dx + a.dy * b.dy + a.dz * b.dz

    dot = dot_product(vector, proj)

    unit = proj / proj.mag

    along = dot / proj.mag * unit
    normal = vector - along

    along['mag'] = np.sqrt(dot_product(along, along))
    normal['mag'] = np.sqrt(dot_product(normal, normal))

    if kind == 'along':
        return along
    elif kind == 'normal':
        return normal
    elif kind is None:
        return (along, normal)

    return (along, normal)


def estimate_clim_gradients(clim):
    ''' Given either argo or ecco climatology, estimate mean gradients
        in along-isopycnal and diapycnal directions'''

    dT = wrap_gradient(clim.Tmean)
    dS = wrap_gradient(clim.Smean)
    dρ = wrap_gradient(clim.ρmean)

    # projection normal to density gradient dρ is along-density surface
    clim['dTiso'] = project_vector(dT, dρ, 'normal').mag
    clim['dTiso'].attrs['long_name'] = 'Along-isopycnal ∇T'
    clim['dTdia'] = project_vector(dT, dρ, 'along').mag
    clim['dTdia'].attrs['long_name'] = 'Diapycnal ∇T'

    clim['dSiso'] = project_vector(dS, dρ, 'normal').mag
    clim['dSiso'].attrs['long_name'] = 'Along-isopycnal ∇S'
    clim['dSdia'] = project_vector(dS, dρ, 'along').mag
    clim['dSdia'].attrs['long_name'] = 'Diapycnal ∇S'

    clim['dTdz'] = dT.dz


def to_density_space(da, rhonew=None):
    ''' Converts a transect *Dataset* to density space
        with density co-ordinate rhonew

        Inputs
        ======
            da : transect DataArray
            rhonew : new density co-ordinate

        Output
        ======
        DataArray with variables interpolated along density co-ordinate.
    '''

    to_da = False
    if isinstance(da, xr.DataArray):
        to_da = True
        da.name = 'my_temp_name'
        da = da.to_dataset()

    if rhonew is None:
        rhonew = np.linspace(da.rho.min(), da.rho.max(), 10)

    def convert_variable(var):
        if var.ndim == 1:
            var = var.expand_dims(['cast'])
            var['rho'] = var.rho.expand_dims(['cast'])

        itemp = np.ones((len(rhonew), len(var.cast)))*np.nan

        for cc, _ in enumerate(var.cast):
            itemp[:, cc] = np.interp(rhonew,
                                     da.rho.isel(cast=cc),
                                     var.isel(cast=cc))

        return (xr.DataArray(itemp,
                             dims=['rho', 'cast'],
                             coords={'rho': rhonew,
                                     'cast': da.cast.values},
                             name=var.name))

    in_dens = xr.Dataset()

    for vv in da.variables:
        if vv in da.coords or vv == 'rho':
            continue

        in_dens = xr.merge([in_dens, convert_variable(da[vv])])

    if 'P' in da.coords:
        Pmat = xr.broadcast(da['P'], da['rho'])[0]
        in_dens = xr.merge([in_dens, convert_variable(Pmat)])

    if 'dist' in da:
        in_dens['dist'] = da.dist
        in_dens = in_dens.set_coords('dist')

    if to_da:
        in_dens = in_dens.set_coords('P').my_temp_name

    return in_dens


def to_depth_space(da, Pold=None, Pnew=None):

    if isinstance(da, xr.DataArray):
        da.name = 'temp'
        da = da.to_dataset()

    if Pold is None:
        Pold = da['P']

    if Pnew is None:
        Pnew = np.linspace(Pold.min(), Pold.max(), 100)

    def convert_variable(var):

        data = np.zeros((len(var.cast), len(Pnew))).T
        for cc, _ in enumerate(var.cast):
            data[:, cc] = np.interp(Pnew,
                                    Pold.isel(cast=cc),
                                    var.isel(cast=cc))

        out = xr.DataArray(data, dims=['P', 'cast'],
                           coords={'P': Pnew, 'cast': da.cast})
        out.coords['cast'] = da.cast
        out.name = var.name

        return out

    in_depth = xr.Dataset()
    for vv in da.variables:
        if vv in da.coords or vv == 'P':
            continue

        in_depth = xr.merge([in_depth, convert_variable(da[vv])])

    if 'rho' in da.coords:
        rmat = xr.broadcast(da['rho'], Pold)[0]
        in_depth = xr.merge([in_depth, convert_variable(rmat)])

    return in_depth.set_coords('rho').temp


def xgradient(da, dim=None, **kwargs):

    if dim is None:
        axis = None
        coords_list = [da.coords[dd].values for dd in da.dims]

    else:
        axis = da.get_axis_num(dim)
        coords_list = [da.coords[dim]]

    grads = np.gradient(da.values, *coords_list, axis=axis, **kwargs)

    if dim is None:
        dda = xr.Dataset()
        for idx, gg in enumerate(grads):
            if da.name is not None:
                name = '∂'+da.name+'/∂'+da.dims[idx]
            else:
                name = '∂/∂'+da.dims[idx]

            dda[name] = xr.DataArray(gg, dims=da.dims, coords=da.coords)
    else:
        if da.name is not None:
            name = '∂'+da.name+'/∂'+da.dims[axis]
        else:
            name = '∂/∂'+da.dims[axis]

        dda = xr.DataArray(grads, dims=da.dims,
                           coords=da.coords, name=name)

    return dda


def smooth_cubic_spline(invar, debug=False):

    # http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
    def moving_average(series):
        b = sp.signal.get_window(('gaussian', 4), 11, fftbins=False)
        average = sp.ndimage.convolve1d(series, b/b.sum())
        var = sp.ndimage.convolve1d(np.power(series-average, 2), b/b.sum())
        return average, var

    # need to use distance co-ordinate because casts need not be evenly spaced
    distnew = invar.dist

    smooth = np.ones((len(invar.rho), len(distnew))) * np.nan
    for idx, rr in enumerate(invar.rho):
        Tvec = invar.sel(rho=rr)
        mask = ~np.isnan(Tvec)

        if len(Tvec[mask]) < 5:
            continue

        _, var = moving_average(Tvec[mask])

        spline = sp.interpolate.UnivariateSpline(
            Tvec.dist[mask], Tvec[mask], k=3, w=1/np.sqrt(var), ext='const')

        Tnew = spline(distnew)

        Tnew[distnew.values < Tvec.dist[mask].min().values] = np.nan
        Tnew[distnew.values > Tvec.dist[mask].max().values] = np.nan

        # plt.figure()
        # plt.plot(Tvec.dist, Tvec, 'o')
        # plt.title('{0:0.2f}'.format(rr.values))
        # plt.plot(distnew, Tnew, 'k-')

        smooth[idx, :] = Tnew

    smooth[np.isnan(invar.values)] = np.nan

    smooth = xr.DataArray(smooth,
                          dims=['rho', 'cast'],
                          coords={'rho': invar.rho, 'cast': invar.cast})

    smooth.coords['dist'] = invar.dist

    if debug:
        invar.plot.contour(levels=50, colors='r')
        smooth.plot.contour(levels=50, colors='k', yincrease=False)

    return smooth


def calc_iso_dia_gradients(field, pres):

    cast_to_dist = False

    if 'dist' not in field.dims and 'cast' in field.dims:
        field = exchange(field.copy(), {'cast': 'dist'})
        pres = exchange(pres.copy(), {'cast': 'dist'})
        cast_to_dist = True

    iso = (ed.xgradient(field, 'dist')
             .interp({'dist': field.dist.values})/1000)
    iso.name = 'iso'

    dia = xr.ones_like(iso) * np.nan
    dia.name = 'dia'

    for idx, dd in enumerate(iso.dist.values[:-1]):
        Tdens_prof = field.sel(dist=dd, method='nearest')
        Pdens_prof = pres.sel(dist=dd)

        dia[:, idx] = np.gradient(Tdens_prof, -Pdens_prof)

    if cast_to_dist:
        iso = exchange(iso, {'dist': 'cast'})
        dia = exchange(dia, {'dist': 'cast'})

    return iso, dia
