import numpy as np
import xarray as xr

import sys
sys.path.append('../')

import eddydiff as ed


def test_circle_project_gradient(debug=False):
    p = np.linspace(0, 150, 150)
    lat = np.linspace(0, 100, 200)
    lon = np.linspace(0, 100, 5)

    dims = ['lon', 'lat', 'pres']
    coords = {'pres': p, 'lat': lat, 'lon': lon}

    pmat, lmat = np.meshgrid(p, lat)
    r = np.sqrt((pmat-60)**2 + (lmat-40)**2)

    circle = (xr.DataArray(np.tile(np.exp(-(r/20)**2), (len(lon), 1, 1)),
                           dims=dims, coords=coords)
              .transpose('pres', 'lat', 'lon'))

    pda = xr.broadcast(circle.pres, circle)[0]

    dc = ed.wrap_gradient(circle)
    dp = ed.wrap_gradient(pda)

    dia, iso = ed.project_vector(dc, dp)

    np.testing.assert_almost_equal(dia.dx.values, np.zeros_like(dia.dx.values))
    np.testing.assert_almost_equal(dia.dy.values, np.zeros_like(dia.dy.values))
    np.testing.assert_almost_equal(iso.dx.values, np.zeros_like(iso.dx.values))
    np.testing.assert_almost_equal(iso.dz.values, np.zeros_like(iso.dz.values))

    if debug is True:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        iso.dy.isel(lon=1).plot(cmap=mpl.cm.RdYlBu_r, ax=ax[0], center=0)
        circle.isel(lon=1).plot.contour(ax=ax[0])
        pda.isel(lon=1).plot.contour(ax=ax[0], levels=10, colors='k')

        dia.dz.isel(lon=1).plot(cmap=mpl.cm.RdYlBu_r, ax=ax[1], center=0)
        circle.isel(lon=1).plot.contour(ax=ax[1])
        pda.isel(lon=1).plot.contour(ax=ax[1], levels=10, colors='k')
        ax[0].set_ylim([150, 0])


def test_project_vectors():
    # some tests
    nx = 10
    ny = 20
    nz = 30

    a = xr.DataArray(np.reshape(np.arange(nx*ny*nz), [nz, ny, nx]),
                     dims=['pres', 'lat', 'lon'],
                     coords={'pres': np.arange(nz),
                             'lat': np.arange(ny),
                             'lon': np.arange(nx)})

    # b = xr.DataArray(np.reshape(np.arange(nx*ny*nz, 0, -1), [nz, ny, nx]),
    #                  dims=['pres', 'lat', 'lon'],
    #                  coords={'pres': np.arange(nz),
    #                          'lat': np.arange(ny),
    #                          'lon': np.arange(nx)})

    da = ed.wrap_gradient(a)
    # db = ed.wrap_gradient(b)

    # projection along gradient should be magnitude
    xr.testing.assert_allclose(ed.project_vector(da, da, 'along').mag,
                               da.mag)

    # projection normal to gradient should be zero
    np.testing.assert_almost_equal(
        ed.project_vector(da, da, 'normal').mag.values,
        np.zeros_like(da.dx))

    dx = xr.Dataset()
    dx['dx'] = xr.ones_like(da.dx)
    dx['dy'] = xr.zeros_like(da.dx)
    dx['dz'] = xr.zeros_like(da.dx)
    dx['mag'] = xr.ones_like(da.dx)

    dy = xr.Dataset()
    dy['dx'] = xr.zeros_like(da.dx)
    dy['dy'] = xr.ones_like(da.dx)
    dy['dz'] = xr.zeros_like(da.dx)
    dy['mag'] = xr.ones_like(da.dx)

    # perpendicular vectors don't project along each other
    xr.testing.assert_allclose(ed.project_vector(dx, dy, 'normal').mag,
                               xr.ones_like(da.dx))
    xr.testing.assert_allclose(ed.project_vector(dx, dy, 'along').mag,
                               xr.zeros_like(da.dx))
