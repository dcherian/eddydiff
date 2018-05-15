import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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
