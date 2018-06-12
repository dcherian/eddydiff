import numpy as np
import xarray as xr


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
    ''' Converts a transect DataArray to density space with density co-ordinate rhonew

        Inputs
        ======
            da : transect DataArray
            rhonew : new density co-ordinate

        Output
        ======
        DataArray with variables interpolated along density co-ordinate.
    '''

    if rhonew is None:
        rhonew = np.linspace(da.rho.min(), da.rho.max(), 30)

    def convert_variable(var):
        itemp = np.ones((len(rhonew), len(da.cast)))*np.nan

        for cc, _ in enumerate(da.cast):
            itemp[:, cc] = np.interp(
                rhonew, da.rho[:, cc], var[:, cc])

        return (xr.DataArray(itemp, dims=['rho', 'dist'],
                             coords={'rho': rhonew,
                                     'dist': da.dist.values},
                             name=var.name))

    in_dens = xr.Dataset()

    for vv in da.variables:
        if vv in da.coords or vv == 'rho':
            continue

        in_dens = xr.merge([in_dens, convert_variable(da[vv])])

        Pmat = xr.broadcast(da['P'], da['rho'])[0]
    if 'P' in da.coords:

        in_dens = xr.merge([in_dens, convert_variable(Pmat)])

    return in_dens


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
