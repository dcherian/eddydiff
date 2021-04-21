from typing import Iterable

import cf_xarray as cfxr
import gsw
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
import seawater as sw
import xgcm

import xarray as xr


def intervals_to_bounds(da):
    assert da.ndim == 1
    name = da.dims[0][:-5]
    vertex = [interval.left for interval in da.data]
    vertex.append(da.data[-1].right)
    bounds = cfxr.vertices_to_bounds(
        xr.DataArray(vertex, dims=f"{name}_vertex"), out_dims=("bounds", "gamma_n")
    )
    bounds.name = f"{name}_bounds"
    bounds.coords[f"{name}"] = bounds.mean("bounds").assign_attrs(bounds=bounds.name)
    return bounds


def intervals_from_vertex(vertex):
    assert vertex.ndim == 1
    name = f"{vertex.dims[0][:-7]}_bins"
    data = [
        pd.Interval(left, right)
        for left, right in zip(vertex.data[:-1], vertex.data[1:])
    ]
    return xr.DataArray(data, dims=name, name=name, coords={name: data})


def exchange(input, kwargs):

    for kk in kwargs.keys():
        d1 = input[kk]
        d2 = input[kwargs[kk]]

        output = input.copy().drop([d1.name, d2.name]).rename({d1.name: d2.name})

    output.coords[d2.name] = d2.values
    output.coords[d1.name] = xr.DataArray(
        d1.values, dims=[d2.name], coords={d2.name: d2.values}
    )

    return output


def format_ecco2(invar):
    mapper = {720: "lon", 360: "lat", 50: "pres", 12: "dayofyear"}
    renamer = {k: mapper[v] for k, v in invar.sizes.items()}
    output = invar.copy(deep=True).drop(["lon", "lat", "dep", "tim"], errors="ignore")
    for name, var in output.items():
        output[name] = var.rename(renamer)

    output = output.drop_dims(renamer.keys(), errors="ignore")
    lon = invar.lon[0, :].values
    lon[lon < 0] += 360
    output["lat"] = invar.lat[:, 0].values
    output["pres"] = invar.dep.values

    output["lon"] = lon
    return output.sortby("lon")


def format_ecco(input):

    dims = input.dims

    if input.ndim == 4:
        output = input.rename({dims[0]: "time"})
    else:
        output = input.copy(deep=True)

    zname = input.dep.dims[0]
    output = output.rename({"dep": "z"}).drop(["lon", "lat"]).rename({zname: "pres"})
    output["pres"] = output["z"]
    output["i4"] = input["lon"].values[0, :].squeeze()
    output["i3"] = input["lat"].values[:, 0].squeeze()

    output = (
        output.rename({"tim": "dayofyear", "i4": "lon", "i3": "lat"})
        .drop(["z"])
        .set_coords(["lon", "lat"])
    )

    return output


def gradient(input):
    """Given an input DataArray, calculate gradients in three directions
    and return it.

    Output
    ======

    dx, dy, dz = gradients in x,y,z
    """

    gradients = np.gradient(input, *[input.coords[dim] for dim in input.dims])

    grads = xr.Dataset()
    for idx, dim in enumerate(input.dims):
        grads["d" + dim] = xr.DataArray(
            gradients[idx], dims=input.dims, coords=input.coords
        )

    grads["mag"] = xr.zeros_like(grads.data_vars["d" + dim])
    for var in grads.data_vars:
        if var == "mag":
            continue

        grads["mag"] += grads[var] ** 2

    grads["mag"] = np.sqrt(grads["mag"])

    return grads


def wrap_gradient(invar):
    """Given an invar DataArray, calculate gradients in three directions
    and return it.

    Output
    ======

    dx, dy, dz = gradients in x,y,z
    """

    nans = np.full((invar.sizes["lat"], invar.sizes["lon"]), np.nan)
    # if invar.ndim == 4:
    #     axis = [1, 2, 3]
    #     nans = np.full_like(invar.isel(time=1, pres=1), np.nan)

    # else:
    #     axis = [0, 1, 2]
    #     nans = np.full_like(invar.isel(pres=1), np.nan)

    gradients = np.gradient(
        invar.data, axis=[invar.get_axis_num(dim) for dim in ["pres", "lat", "lon"]]
    )

    def make_diff(invar, name):
        dlat = invar[name].diff(name, label="upper")
        return xr.concat(
            [
                xr.DataArray(
                    [dlat[0]], dims=dlat.dims, coords={name: [invar[name][0]]}
                ),
                dlat,
            ],
            dim=name,
        )

    dims = invar.dims
    coords = invar.coords

    dlon = xr.DataArray(
        nans,
        dims=["lat", "lon"],
        coords={"lat": invar.lat, "lon": invar.lon},
        name="Δlon",
    )
    for ll, lat in enumerate(invar.lat):
        dlon.values[ll, 1:] = gsw.distance(
            lat=xr.broadcast(lat, invar.lon)[0].values, lon=invar.lon.values, p=[0]
        )
    dlon[:, 0] = dlon[:, 1]

    grads = xr.Dataset()
    grads["dz"] = (
        xr.DataArray(gradients[0], dims=dims, coords=coords)
        / make_diff(invar, "pres")
        * -1
    )
    # Δlat is roughly 111km everywhere
    grads["dy"] = (
        xr.DataArray(gradients[1], dims=dims, coords=coords)
        / make_diff(invar, "lat")
        / 1.11e5
    )
    grads["dx"] = xr.DataArray(gradients[2], dims=dims, coords=coords) / dlon

    grads["mag"] = np.sqrt(grads.dx ** 2 + grads.dy ** 2 + grads.dz ** 2)

    return grads


def project_vector(vector, proj, kind=None):
    """Project 'vector' along 'proj'.

    Returns
    =======

    List of DataArrays (vector projection, rejection)
    """

    def dot_product(a, b):
        """ dot product of 2 gradient vectors """
        return a.dx * b.dx + a.dy * b.dy + a.dz * b.dz

    dot = dot_product(vector, proj)

    unit = proj / proj.mag

    along = dot / proj.mag * unit
    normal = vector - along

    along["mag"] = np.sqrt(dot_product(along, along))
    normal["mag"] = np.sqrt(dot_product(normal, normal))

    if kind == "along":
        return along
    elif kind == "normal":
        return normal
    elif kind is None:
        return (along, normal)

    return (along, normal)


def estimate_clim_gradients(clim):
    """Given either argo or ecco climatology, estimate mean gradients
    in along-isopycnal and diapycnal directions"""

    dT = wrap_gradient(clim.Tmean)
    dS = wrap_gradient(clim.Smean)
    dρ = wrap_gradient(clim.ρmean)

    # projection normal to density gradient dρ is along-density surface
    clim["dTiso"] = project_vector(dT, dρ, "normal").mag
    clim["dTiso"].attrs["long_name"] = "Along-isopycnal ∇T"
    clim["dTdia"] = project_vector(dT, dρ, "along").mag
    clim["dTdia"].attrs["long_name"] = "Diapycnal ∇T"

    clim["dSiso"] = project_vector(dS, dρ, "normal").mag
    clim["dSiso"].attrs["long_name"] = "Along-isopycnal ∇S"
    clim["dSdia"] = project_vector(dS, dρ, "along").mag
    clim["dSdia"].attrs["long_name"] = "Diapycnal ∇S"

    clim["dTdz"] = dT.dz
    clim["dTdz"].attrs["long_name"] = "dT/dz"
    clim["dSdz"] = dS.dz
    clim["dSdz"].attrs["long_name"] = "dS/dz"


def to_density_space(da, rhonew=None):
    """Converts a transect *Dataset* to density space
    with density co-ordinate rhonew *by interpolation*

    Inputs
    ======
        da : transect DataArray
        rhonew : new density co-ordinate

    Output
    ======
    DataArray with variables interpolated along density co-ordinate.
    """

    to_da = False
    if isinstance(da, xr.DataArray):
        to_da = True
        da.name = "my_temp_name"
        da = da.to_dataset()

    if rhonew is None:
        rhonew = np.linspace(da.rho.min(), da.rho.max(), 10)

    def convert_variable(var):
        if var.ndim == 1:
            var = var.expand_dims(["cast"])
            var["rho"] = var.rho.expand_dims(["cast"])

        itemp = np.ones((len(rhonew), len(var.cast))) * np.nan

        for cc, _ in enumerate(var.cast):
            itemp[:, cc] = np.interp(
                rhonew,
                da.rho.isel(cast=cc),
                var.isel(cast=cc),
                left=np.nan,
                right=np.nan,
            )

        return xr.DataArray(
            itemp,
            dims=["rho", "cast"],
            coords={"rho": rhonew, "cast": da.cast.values},
            name=var.name,
        )

    in_dens = xr.Dataset()

    for vv in da.variables:
        if vv in da.coords or vv == "rho":
            continue

        in_dens = xr.merge([in_dens, convert_variable(da[vv])])

    if "pres" in da:
        in_dens = xr.merge([in_dens, convert_variable(da["pres"])])
        in_dens = in_dens.rename({"pres": "P"})
    elif "P" in da.coords:
        Pmat = xr.broadcast(da["P"], da["rho"])[0]
        in_dens = xr.merge([in_dens, convert_variable(Pmat)])

    if "dist" in da:
        in_dens["dist"] = da.dist
        in_dens = in_dens.set_coords("dist")

    if to_da:
        in_dens = in_dens.set_coords("P").my_temp_name

    return in_dens


def to_depth_space(da, Pold=None, Pnew=None):

    if isinstance(da, xr.DataArray):
        da.name = "temp"
        da = da.to_dataset()

    if Pold is None:
        Pold = da["P"]

    if Pnew is None:
        Pnew = np.linspace(Pold.min(), Pold.max(), 100)

    def convert_variable(var):

        data = np.zeros((len(var.cast), len(Pnew))).T
        for cc, _ in enumerate(var.cast):
            data[:, cc] = np.interp(
                Pnew, Pold.isel(cast=cc), var.isel(cast=cc), left=np.nan, right=np.nan
            )

        out = xr.DataArray(
            data, dims=["P", "cast"], coords={"P": Pnew, "cast": da.cast}
        )
        out.coords["cast"] = da.cast
        out.name = var.name

        return out

    in_depth = xr.Dataset()
    for vv in da.variables:
        if vv in da.coords or vv == "P":
            continue

        in_depth = xr.merge([in_depth, convert_variable(da[vv])])

    if "rho" in da.coords:
        rmat = xr.broadcast(da["rho"], Pold)[0]
        in_depth = xr.merge([in_depth, convert_variable(rmat)])

    return in_depth.set_coords("rho").temp


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
            # if da.name is not None:
            #     name = 'd'+da.name+'d'+da.dims[idx]
            # else:
            name = "d" + da.dims[idx]

            dda[name] = xr.DataArray(gg, dims=da.dims, coords=da.coords)
    else:
        # if da.name is not None:
        #     name = '∂'+da.name+'/∂'+da.dims[axis]
        # else:
        name = "d" + da.dims[axis]

        dda = xr.DataArray(grads, dims=da.dims, coords=da.coords, name=name)

    return dda


def fit_spline(x, y, k=3, ext="const", debug=False, **kwargs):

    # http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
    def moving_average(series):
        b = sp.signal.get_window(("gaussian", 4), 11, fftbins=False)
        average = sp.ndimage.convolve1d(series, b / b.sum())
        var = sp.ndimage.convolve1d(np.power(series - average, 2), b / b.sum())
        return average, var

    average, var = moving_average(y)

    w = 1 / np.sqrt(var)

    mask = ~np.isnan(y)
    # w[0:2] = w[0:2]/1.5
    # w[-2:] = w[-2:]/1.5

    # spline = sp.interpolate.make_interp_spline(x, y, bc_type='natural')
    spline = sp.interpolate.UnivariateSpline(
        x[mask], y[mask], k=k, w=w[mask], ext=ext, **kwargs
    )

    vals = np.full_like(x, np.nan)
    vals[mask] = spline(x[mask])

    if isinstance(y, xr.DataArray):
        vals = xr.DataArray(vals, dims=y.dims, coords=y.coords)

    if debug:
        plt.figure()
        plt.plot(x, y, "o")
        plt.plot(x, vals)
        plt.legend(["raw", "smoothed"])

    return spline, vals


def smooth_cubic_spline(invar, debug=False):

    # need to use distance co-ordinate because casts need not be evenly spaced
    distnew = invar.dist

    smooth = np.ones((len(invar.rho), len(distnew))) * np.nan
    for idx, rr in enumerate(invar.rho):
        Tvec = invar.sel(rho=rr)
        mask = ~np.isnan(Tvec)

        if len(Tvec[mask]) < 5:
            continue

        spline, _ = fit_spline(Tvec.dist[mask], Tvec[mask], k=4)

        Tnew = spline(distnew)

        Tnew[distnew.values < Tvec.dist[mask].min().values] = np.nan
        Tnew[distnew.values > Tvec.dist[mask].max().values] = np.nan

        # plt.figure()
        # plt.plot(Tvec.dist, Tvec, 'o')
        # plt.title('{0:0.2f}'.format(rr.values))
        # plt.plot(distnew, Tnew, 'k-')

        smooth[idx, :] = Tnew

    if invar.shape != smooth.shape:
        smooth = smooth.transpose()

    smooth[np.isnan(invar.values)] = np.nan

    smooth = xr.DataArray(smooth, dims=invar.dims, coords=invar.coords)

    smooth.coords["dist"] = invar.dist

    if debug:
        plt.figure()
        invar.plot.contour(y="rho", levels=50, colors="r")
        smooth.plot.contour(y="rho", levels=50, colors="k", yincrease=False)

    return smooth


def calc_iso_dia_gradients(field, pres, debug=False):

    cast_to_dist = False

    if "dist" not in field.dims and "cast" in field.dims:
        field = exchange(field.copy(), {"cast": "dist"})
        pres = exchange(pres.copy(), {"cast": "dist"})
        cast_to_dist = True

    iso = xgradient(field, "dist").interp({"dist": field.dist.values}) / 1000
    iso.name = "iso"

    dia = xr.ones_like(iso) * np.nan
    dia.name = "dia"

    for idx, dd in enumerate(iso.dist.values):
        Tdens_prof = field.sel(dist=dd, method="nearest")
        Pdens_prof = pres.sel(dist=dd)

        if dia.dims[1] == "dist":
            dia[:, idx] = np.gradient(Tdens_prof, -Pdens_prof)
        else:
            dia[idx, :] = np.gradient(Tdens_prof, -Pdens_prof)

    if cast_to_dist:
        iso = exchange(iso, {"dist": "cast"})
        dia = exchange(dia, {"dist": "cast"})

    iso.attrs["name"] = "Isopycnal ∇"
    dia.attrs["name"] = "Diapycnal ∇"

    if debug:
        f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        f.set_size_inches((10, 5))
        iso.plot(ax=ax[0], x="cast", yincrease=False)
        dia.plot(ax=ax[1], x="cast", yincrease=False)
        ax[1].set_ylabel("")
        plt.tight_layout()

    return iso, dia


def bin_avg_in_density(input: pd.DataFrame, ρbins: Iterable, dname="cast"):
    """
    Takes input dataframe for transect, bins each profile by density
    and averages. Returns average data as function of transect distance,
    mean density in bin.

    """

    return (
        input.groupby([dname, pd.cut(input.rho, ρbins)])
        .mean()
        .rename({"rho": "mean_rho"}, axis=1)
        .reset_index()
        .drop("rho", axis=1)
        .rename({"mean_rho": "rho"}, axis=1)
        .dropna(how="any")
        .set_index([dname, "rho"])
        .to_xarray(sparse=True)
    )


def bin_avg_in_density_time(chisub: xr.Dataset, bins: dict, strftime="%Y-%m"):
    """
    Takes input dataset and bin averages.

    Written for TAO χpod analysis.
    """

    assert len(bins) == 1
    var, ρbins = next(iter(bins.items()))
    binvar = chisub.cf[var].name

    df = chisub.compute().to_dataframe().dropna()
    df = df.reset_index()
    time_grouper = df.time.dt.strftime(strftime)

    grouped = df.groupby([pd.cut(df[binvar], ρbins), time_grouper])

    result = grouped.mean()
    result["numobs"] = grouped.size()
    result = result.dropna(how="any").drop(binvar, axis=1).reset_index()
    result["time"] = pd.to_datetime(result["time"])

    binned = result.set_index([binvar, "time"])

    pdentime = binned.index.to_frame()
    midpoints = pd.Series([v.mid for v in pdentime[binvar].values], name=binvar)
    newindex = pd.MultiIndex.from_arrays([midpoints, pdentime["time"]])
    float_indexed = binned.set_index(newindex)
    chidens = xr.Dataset.from_dataframe(float_indexed)
    # chidens = chidens.rename({"pden": "pden"})

    chidens["chi"].attrs["long_name"] = "$χ$"
    chidens["KtTz"].attrs["long_name"] = "$⟨K_T T_z⟩$"

    return chidens


def plot_bar_Ke(Ke, dTdz_log=True, Ke_log=True, cole=None):

    # if cole is None:
    #     cole = read_cole()

    f, ax = plt.subplots(2, 3, sharey=True)
    f.set_size_inches(8, 6)
    try:
        f.suptitle(Ke.name, y=1.01)
    except AttributeError:
        pass

    (Ke.KT).plot.barh(x="rho", log=True, ax=ax[0, 0], title="$K_T$")
    (Ke.dTdz).plot.barh(
        x="rho", log=dTdz_log, ax=ax[0, 1], title="$T_z (color), T^m_z (black)$"
    )
    (
        (Ke.dTmdz).plot.barh(
            x="rho", log=dTdz_log, ax=ax[0, 1], edgecolor="black", color="none"
        )
    )

    ((Ke.KtTz * 1025 * 4200)).plot.barh(
        x="rho", ax=ax[0, 2], title="$ρ c_p ⟨K_T T_z⟩ = ⟨J_q⟩$"
    )

    # ((Ke.dTmdz).plot.barh(x='rho', log=False,
    #                       ax=ax[1, 0], title='$T_z^m, T_z$'))
    # ((Ke.dTdz)).plot.barh(x='rho', log=False,
    #                       ax=ax[1, 0], color='none', edgecolor='black')

    (
        np.abs(Ke.KtTz * Ke.dTmdz).plot.barh(
            x="rho",
            log=True,
            ax=ax[1, 0],
            title="$|⟨K_T T_z⟩ T^m_z| (color), χ/2 (black)$",
        )
    )
    (
        (Ke.chi / 2).plot.barh(
            x="rho", log=True, ax=ax[1, 0], color="none", edgecolor="black"
        )
    )

    (
        (Ke.dTiso).plot.barh(
            x="rho", log=True, ax=ax[1, 1], title="$dT_{iso}$ (log scale)"
        )
    )

    (Ke.Ke.where(Ke.Ke > 0).plot.barh(x="rho", log=Ke_log, ax=ax[1, 2], title="$K_e$"))
    np.abs(Ke.Ke.where(Ke.Ke < 0)).plot.barh(
        x="rho", log=Ke_log, ax=ax[1, 2], title="$K_e$", edgecolor="black", color="none"
    )

    plt.gca().invert_yaxis()

    def fix_axes(aa):
        aa.xaxis.tick_top()
        aa.spines["top"].set_visible(True)
        aa.spines["bottom"].set_visible(False)

    # [fix_axes(aa) for aa in ax.flat]

    plt.tight_layout()


def plot_transect_Ke(transKe):
    if "cast" in transKe.chi.dims:
        dname = "cast"
    else:
        dname = dname

    f, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    f.set_size_inches(10, 14)

    np.log10(transKe.chi / 2).plot(ax=ax[0, 0], x=dname, cmap=mpl.cm.Reds)
    np.log10(transKe.KtTz).plot(ax=ax[1, 0], x=dname, cmap=mpl.cm.Reds)
    (transKe.dTmdz).plot(ax=ax[0, 1], x=dname)
    (transKe.dTdz).plot(ax=ax[1, 1], x=dname)
    (transKe.dTiso).plot(ax=ax[2, 0], x=dname)
    ((np.sign(transKe.Ke) * np.log10(np.abs(transKe.Ke))).plot(ax=ax[2, 1], x=dname))
    ax[0, 0].set_ylim([1027, 1019])
    plt.tight_layout()


def read_cole():
    cole = (
        xr.open_dataset(
            "../datasets/argo-diffusivity/" + "ArgoTS_eddydiffusivity_20052015_1deg.nc",
        )
        .rename(
            {"latitude": "lat", "longitude": "lon", "density": "sigma", "depth": "pres"}
        )
        .set_coords(
            [
                "lat",
                "lon",
                "sigma",
                "start_year",
                "end_year",
                "mixing_efficiency",
                "minimum_points",
                "maximum_mixing_length",
            ]
        )
    )

    cole = cole.cf.guess_coord_axis()
    cole["sigma"] = cole.sigma.data
    cole["pres"] = cole.pres.data
    del cole.minimum_points.attrs["axis"]
    del cole.minimum_points.attrs["standard_name"]

    long_names = {
        "salinity_gradient": "$|∇S|$",
        "mixing_length": "$λ$",
        "salinity_mean": "$S$",
        "salinity_std": "$\sqrt{S'S'}$",  # noqa
        "diffusivity": "$κ$",
    }

    for var in cole:
        for key, long_name in long_names.items():
            if key in var:
                cole[var].attrs["description"] = cole[var].attrs["long_name"]
                cole[var].attrs["long_name"] = long_name
                units = cole[var].attrs["units"]
                if "absolute salinity, " in units.lower():
                    cole[var].attrs["units"] = units[18:]
    # cole['rho'] = sw.dens(cole)
    cole["diffusivity_first"] = cole.diffusivity.bfill(dim="pres").isel(pres=0)
    cole.sigma.attrs["positive"] = "down"
    cole.pres.attrs["positive"] = "down"
    return cole


def get_region_from_transect(transect):
    return {
        "lon": slice(transect.lon.min(), transect.lon.max()),
        "lat": slice(transect.lat.min(), transect.lat.max()),
        "pres": slice(transect.pres.min(), np.max([300, transect.pres.max()])),
    }


def convert_mat_to_netcdf():

    tr1 = sp.io.loadmat("../datasets/bob-ctd-chipod/transect_1.mat", squeeze_me=True)
    tr2 = sp.io.loadmat("../datasets/bob-ctd-chipod/transect_2.mat", squeeze_me=True)
    tr3 = sp.io.loadmat("../datasets/bob-ctd-chipod/transect_3.mat", squeeze_me=True)

    tr1.keys()

    # convert mat to xarray to netcdf
    for idx, tr in enumerate([tr1, tr2, tr3]):

        coords = {
            "cast": np.arange(tr["P"].shape[1]),
            "P": np.arange(tr["P"].shape[0]),
            "pres": (["P", "cast"], tr["P"]),
            "lon": (["cast"], tr["lon"]),
            "lat": (["cast"], tr["lat"]),
            "dist": (["cast"], tr["dist"]),
        }

        transect = xr.Dataset()
        transect = xr.merge(
            [
                transect,
                xr.Dataset(
                    {
                        "chi": (["P", "cast"], tr["chi"]),
                        "eps": (["P", "cast"], tr["eps"]),
                        "KT": (["P", "cast"], tr["KT"]),
                        "dTdz": (["P", "cast"], tr["dTdz"]),
                        "N2": (["P", "cast"], tr["N2"]),
                        "fspd": (["P", "cast"], tr["fspd"]),
                        "T": (["P", "cast"], tr["T"]),
                        "S": (["P", "cast"], tr["S"]),
                        "theta": (["P", "cast"], tr["theta"]),
                        "sigma": (["P", "cast"], tr["sigma"]),
                    },
                    coords=coords,
                ),
            ]
        )
        transect["rho"] = xr.DataArray(
            sw.dens(transect["S"], transect["T"], transect["pres"]),
            dims=transect["T"].dims,
            coords=coords,
        )
        transect.dist.attrs["units"] = "km"

        mask2d = np.logical_or(transect["T"].values < 1, transect["S"].values < 1)
        mask1d = np.logical_or(transect.lon < 1, transect.lat < 1)

        for var in transect.variables:
            if var == "cast" or var == "P":
                continue

            if transect[var].ndim == 2:
                transect[var].values[mask2d] = np.nan
            elif transect[var].ndim == 1:
                transect[var].values[mask1d] = np.nan

        transect.to_netcdf(
            "../datasets/bob-ctd-chipod/transect_{0:d}.nc".format(idx + 1)
        )


def average_transect_1d(transect, nbins=10):
    """ Given transect, group by density bins and average in those bins. """

    # TODO: Keep or remove?
    try:
        if isinstance(transect, pd.DataFrame):
            trdf = transect.drop(["fspd", "N2", "theta", "sigma"], axis=1)
        else:
            trdf = transect.drop(["fspd", "N2", "theta", "sigma"]).to_dataframe()
    except ValueError:
        trdf = transect.to_dataframe()

    if "KtTz" not in trdf:
        trdf["KtTz"] = trdf["KT"] * trdf["dTdz"]

    trdf = trdf.reset_index()

    ρinds, ρbins = pd.qcut(trdf.rho, nbins, precision=1, retbins=True)

    # means for transect
    trmean = trdf.groupby(ρinds).mean()

    return trmean, ρbins


def average_clim(clim, transect, ρbins):
    if "lon" in clim.dims:
        region = get_region_from_transect(transect)
        clim = clim.sel(**region)

    clim["Pmean"] = xr.broadcast(clim.pres, clim.Tmean)[0]
    # climrho = clim.groupby_bins(clim.ρmean, ρbins).mean()

    # dTdz_local is the pointwise estimate of dTdz that then
    # gets averaged on ispycnals later.
    # dTdz is the derivative of a mean T profile (averaged on isopycnals)
    clim = clim.rename({"dTdz": "dTdz_local"})

    clim = clim.to_dataframe().reset_index()
    climrho = clim.groupby(pd.cut(clim.ρmean, ρbins, precision=1)).mean()
    # sp, _ = fit_spline(climrho.Pmean, climrho.Tmean)
    # climrho['dTdz'] = -sp.derivative(1)(climrho.Pmean)
    climrho["dTdz"] = np.gradient(climrho.Tmean, -climrho.Pmean)

    # dTiso = clim.dTiso.sel(**region)
    # clim.ρmean.plot.hist(alpha=0.5)
    # transect.rho.plot.hist(alpha=0.5)

    # climdf = clim.to_dataframe().reset_index()
    # ρinds = pd.cut(climdf.ρmean, ρbins)
    # climmean = climdf.groupby(ρinds).mean()
    return climrho


def estimate_Ke(trans, clim):
    Ke = pd.DataFrame()

    Ke["chi"] = trans.chi
    Ke["KtTz"] = trans.KtTz
    Ke["dTdz"] = trans.dTdz
    Ke["dTmdz"] = clim.dTdz  # .where(clim.dTdz < trans.dTdz, trans.dTdz)
    Ke["dTiso"] = clim.dTiso
    Ke["KT"] = trans.KT
    Ke["Ke"] = (Ke.chi / 2 - np.abs(Ke.KtTz * Ke.dTmdz)) / Ke.dTiso ** 2

    return Ke


def compare_means_clim_transect(trmean, clim):
    f, ax = plt.subplots(1, 2, sharey=True)

    ax[0].plot(trmean["T"], trmean["pres"])
    ax[0].plot(clim["Tmean"], clim["Pmean"])
    ax[0].set_ylabel("P")
    ax[0].set_xlabel("T")
    # ax[0].set_xaxis_top()
    ax[0].invert_yaxis()

    ax[1].plot(np.gradient(trmean["T"], -trmean["pres"]), trmean["pres"])
    ax[1].plot(np.gradient(clim["Tmean"], -clim["Pmean"]), clim["Pmean"])
    ax[1].set_xlabel("d$T^m$/dz")
    # ax[1].set_xaxis_top()
    ax[1].invert_yaxis()


def process_transect_1d(transect, clim, name=None, nbins=10):
    if "time" in clim.dims:
        clim = clim.copy().sel(time=transect.time.dt.month.median(), method="nearest")

    trmean, ρbins = average_transect_1d(transect, nbins=nbins)
    gradmean = average_clim(clim, transect, ρbins)

    Ke = estimate_Ke(trmean, gradmean)

    if name is not None:
        Ke.name = name

    return Ke


def transect_to_density_space(transect, nbins=12):

    # 1. Bin average in density space.
    _, bins = pd.cut(transect.rho.values.ravel(), nbins, retbins=True)

    trmean = xr.Dataset()

    # TODO: can I vectorize this?
    for var in transect.data_vars:
        for cc in transect.cast:
            df = transect[var].sel(cast=cc).to_dataframe()
            df["rho"] = transect["rho"].sel(cast=cc)
            dfmean = df.groupby(pd.cut(df.rho, bins)).mean().to_xarray()
            dfmean["rho"] = (bins[:-1] + bins[1:]) / 2
            dfmean = dfmean.drop(["cast", "dist", "lon", "lat"]).expand_dims(["cast"])
            dfmean["cast"] = np.asarray([cc])

            trmean = xr.merge([trmean, dfmean])

    trmean["lon"] = transect.lon.drop(["lat", "lon", "dist"])
    trmean["lat"] = transect.lat.drop(["lat", "lon", "dist"])
    trmean["dist"] = transect.dist.drop(["lat", "lon", "dist"])

    trmean_rho = (
        trmean.set_coords(["dist", "lon", "lat"]).rename({"pres": "P"}).transpose()
    )

    return trmean_rho


def bin_to_density_space(transect, bins=30):
    """ Attempt number 3. """

    if np.isscalar(bins):
        _, bins = pd.qcut(transect.rho.values.ravel(), bins, retbins=True)
    else:
        bins = np.asarray(bins)

    df = transect.to_dataframe().reset_index()

    df["rho_bins"] = pd.cut(
        df.rho, bins, labels=np.round((bins[1:] + bins[:-1]) / 2, 3)
    )

    if "cast" in transect.dims:
        dim = "cast"
    else:
        dim = "time"
    binned = df.groupby([dim, "rho_bins"]).mean()

    # this only works if the index is not pandas.IntervalIndex
    trdens = binned.to_xarray()

    extra_coords = ["dist", "lon", "lat"]
    for var in extra_coords:
        if var in transect:
            trdens[var] = transect.reset_coords()[var]
            trdens = trdens.set_coords(var)

    if "P" in trdens:
        trdens = trdens.set_coords("P")

    trdens = trdens.rename({"rho": "mean_rho", "rho_bins": "rho"})
    trdens["rho"] = trdens.rho.astype(np.float32)
    trdens["rho"].attrs["positive"] = "down"

    return trdens, bins


def read_all_datasets(kind="annual", transect=None):
    """Reads in
    1. Cole et al estimate
    2. ECCO gradients
    3. Argo gradients

    Inputs
    ------
    kind: 'annual' (default) or 'monthly'. If 'annual', averages in time.
    transect: (optional), if provided will subset the data
              to particular transect.
    """

    cole = read_cole()

    aber = xr.open_dataset("../datasets/diffusivity_AM2013.nc")

    name = "monthly_iso_gradients.zarr"
    argograd = xr.open_zarr("../datasets/argo_" + name, decode_times=False)
    # argograd.time.attrs["calendar"] = "360_day"
    # argograd = xr.decode_cf(argograd)
    # argograd = argograd.assign_coords(time=argograd.indexes["time"].to_datetimeindex())

    eccograd = xr.open_zarr("../datasets/ecco_" + name, decode_times=False)

    if kind == "annual":
        eccograd = eccograd.mean(dim="time")
        argograd = argograd.mean(dim="time")

    # if provided with transect, subset!
    if transect is not None:
        # TODO: selecting region before interp doesn't work too well
        # region = get_region_from_transect(transect)

        # pick months for the transect
        month = np.unique(transect.time.dt.month.dropna(dim="cast"))

        interp_args = dict(
            lon=transect.reset_coords().lon,
            lat=transect.reset_coords().lat,
            pres=transect.reset_coords().pres,
        )

        eccograd = eccograd.sel(time=month).mean("time").compute().interp(**interp_args)

        eccograd["sigma_0"] = eccograd["ρmean"]
        eccograd["sigma_0"].values = sw.pden(
            eccograd.Smean,
            eccograd.Tmean,
            xr.broadcast(eccograd.pres, eccograd.Tmean)[0],
            0,
        )

        argograd = argograd.compute().interp(**interp_args).dropna(dim="P", how="all")

        cole = cole.compute().interp(**interp_args).dropna(dim="P", how="all")

        interp_args.pop("pres")
        aber = aber.compute().interp(**interp_args)

    return eccograd, argograd, cole, aber


def average_clim_1d(clim):
    """Takes climatological fields with dimensions (cast, P),
    estimates along-transect, vertical gradients."""
    pass


def process_ecco_gradients():

    from dcpy.oceans import dataset_center_pacific

    ρfile = "../datasets/ecco/interp_climatology/RHOAnoma.0001.nc"
    Sfile = "../datasets/ecco/interp_climatology/SALT.0001.nc"
    Tfile = "../datasets/ecco/interp_climatology/THETA.0001.nc"

    chunks = {"time": 12}

    T = xr.open_dataset(Tfile, decode_times=False).pipe(format_ecco).chunk(chunks)
    S = xr.open_dataset(Sfile, decode_times=False).pipe(format_ecco).chunk(chunks)
    ρ = xr.open_dataset(ρfile, decode_times=False).pipe(format_ecco).chunk(chunks)

    ecc = xr.merge([T, S, ρ]).rename({"THETA": "Tmean", "SALT": "Smean"})

    # roll so that pacific is in the middle and we have coverage of all 3 basins
    monthly = dataset_center_pacific(ecc)

    # in-situ density anomaly!!!
    monthly["ρmean"] = monthly.RHOAnoma + 1029

    annual = monthly.mean(dim="time")

    estimate_clim_gradients(annual)

    monthly.attrs[
        "name"
    ] = "Annual mean fields and isopycnal, diapycnal gradients from ECCO v4r3"
    annual.attrs["dataset"] = "ecco"
    annual.to_netcdf("../datasets/ecco_annual_iso_gradient.nc", compute=True)

    estimate_clim_gradients(monthly)
    monthly.attrs[
        "name"
    ] = "Monthly mean fields and isopycnal, diapycnal gradients from ECCO v4r3"
    monthly.attrs["dataset"] = "ecco"

    from dask.diagnostics import ProgressBar

    delayed = monthly.to_netcdf(
        "../datasets/ecco_monthly_iso_gradients.nc", compute=False
    )

    with ProgressBar():
        delayed.compute(num_workers=2)


def plot_transect_var(
    x, y, data, fill=None, contour=None, bar2=None, xlim=None, xticks=None
):

    f, ax = plt.subplots(1, len(data[x]), sharex=True, sharey=True)

    if ~isinstance(ax, list):
        ax = list(ax)

    f.set_size_inches(14, 7)
    for idx, cc in enumerate(data[x]):
        datavec = data.sel({x: cc})
        if np.all(np.isnan(datavec)) or np.all(datavec[~np.isnan(datavec)] < 0):
            ax[idx].set_visible(False)
            continue

        ax[idx].barh(y=data[y], width=datavec, align="center", color="gray", height=0.3)

        if bar2 is not None:
            ax[idx].barh(
                y=data[y],
                width=bar2.sel({x: cc}),
                align="center",
                color="none",
                edgecolor="w",
                height=0.15,
            )

        ax[idx].set_xscale("log")
        ax[idx].xaxis.tick_top()
        ax[idx].spines["bottom"].set_visible(False)
        ax[idx].spines["top"].set_visible(True)

    ax[idx].set_yticks(data[y])

    # create big background axis
    from matplotlib.transforms import Bbox

    pos1 = ax[0].get_position()
    pos2 = ax[-1].get_position()
    posnew = Bbox([pos1.p0, pos2.p1])
    axback = plt.axes(posnew, zorder=-1, sharey=ax[0])

    titlestr = "bar: " + data.name

    # plot filled and contour variable if possible
    if fill is not None:
        fill.plot.contourf(ax=axback, levels=50, cmap=mpl.cm.RdBu_r, add_colorbar=False)
        fill.plot.contour(ax=axback, levels=[0], colors="k", linestyles="dashed")
        titlestr += " | fill: " + fill.name

    if contour is not None:
        contour.plot.contour(
            ax=axback, colors="k", levels=[10, 25, 50, 75, 100, 150, 200]
        )
        titlestr += " | contour: " + contour.name

    axback.invert_yaxis()
    if xticks is not None:
        ax[-1].set_xticks(xticks)

    if xlim is not None:
        ax[-1].set_xlim(xlim)

    # f.suptitle(titlestr + ' | navg = ' + str(transKe.attrs['navg']), y=0.95)

    return ax, axback


def regrid_to_isopycnals(clim, bins, target_kind="edges"):
    """
    Regrids climatologies to provided isopycnal bins
    """

    from .regrid import remap_full

    isoT = remap_full(
        clim[["Tmean", "pres_bounds"]],
        clim.pden,
        xr.DataArray(bins, dims="target"),
        dim="pres",
        target_kind=target_kind,
    )
    return isoT


def estimate_gradients(grad, bins, bin_var="pden", debug=False):
    """
    Estimates the isopycnal and vertical gradients of temperature

    Parameters
    ----------

    grad: Dataset
        Climatological fields with temperature
    bins: np.array
        Isopycnal bin edges
    bin_var: optional, str
        Variable name in ``grad`` to match ``bins``. Passed to cf_xarray.
        E.g. "pden", "gamma_n"

    Returns
    -------

    Dataset with ``dTdy`` and ``dTdz``

    Notes
    -----

    Procedure:
      - First conservatively regrids T to isopycnal space using bin averages
      - Then fits univariate spline to mean T in each isopycnal bin
      - Estimates ∂T/∂y using this spline fit.
      - ∂T/∂z is estimated using the mean separation between isopycnal bin edges
        (product of the isopycnal regridding code)
    """
    import dcpy.interpolate

    bins = np.asarray(bins)
    centers = (bins[1:] + bins[:-1]) / 2

    grid = xgcm.Grid(
        grad, periodic=False, coords={"Y": {"center": "lat"}, "Z": {"center": "pres"}}
    )

    # isoT = ed.regrid_to_isopycnals(grad, bins).rename({"pden_bins": "pden"})

    # if np.any(bins < 1000):
    #    raise ValueError(f"bins={bins} < 1000")

    target_data = grad.cf[bin_var]
    binned_name = target_data.name

    isoT = xr.Dataset()
    isoT["Tmean"] = grid.transform(
        grad.Tmean,
        "Z",
        target=centers,
        target_data=target_data,
        method="linear",
    ).compute()

    isoT.coords[f"{binned_name}_bounds"] = cfxr.vertices_to_bounds(bins)
    isoT[binned_name].attrs["bounds"] = f"{binned_name}_bounds"

    z_regrid = grid.transform(
        grad.pres.broadcast_like(grad.pden),
        "Z",
        target=bins,
        target_data=target_data,
        method="linear",
    ).compute()

    # calculate the layer thickness of the new coord bounds
    isoT.coords["dz_remapped"] = z_regrid.diff(bin_var).drop_vars(bin_var)
    isoT.coords["pres"] = (
        ((z_regrid + z_regrid.shift({binned_name: -1})) / 2)
        .isel({binned_name: slice(-1)})
        .assign_coords({binned_name: isoT[binned_name].data})
    )
    isoT.coords["pres"].attrs.update({"long_name": "Pressure at bin centers"})

    isoT = isoT.cf.guess_coord_axis()

    # get isopycnal gradients of T by fitting spline
    # to average T in density bins
    isoT.coords["y"] = np.sign(isoT.lat) * dcpy.util.latlon_to_distance(
        isoT.cf["latitude"],
        isoT.cf["longitude"],
        central_lat=0,
        central_lon=isoT.lon.item(),
    )
    isoT.y.attrs["axis"] = "Y"

    sub = isoT.Tmean.chunk({"lat": -1}).swap_dims({"lat": "y"})
    spinterp = dcpy.interpolate.UnivariateSpline(sub, "y")

    isoT["dTdy"] = spinterp.derivative(sub.y).swap_dims({"y": "lat"}).compute()

    if debug:
        plt.figure()
        spinterp.smooth().compute().cf.plot.line(x="lat", hue=bin_var, add_legend=False)
        sub.cf.plot.line(x="lat", hue=bin_var, marker=".", ls="none", add_legend=True)
        plt.suptitle(grad.attrs["dataset"])

    # regrid to the bin edges
    edgeT = (
        grid.transform(
            grad.Tmean, "Z", target=bins, target_data=target_data, method="linear"
        ).compute()
        # ed.regrid_to_isopycnals(grad, bins, "centers")
        # .load()
        # .assign_coords(pden_bins=centers[:-1])
        # .reindex_like(isoT, method="nearest")
        # .assign_coords(pden_bins=centers)
    )

    isoT["dTdz"] = (
        -1
        * edgeT.cf.diff(bin_var).assign_coords({bin_var: isoT.cf[bin_var].values})
        / isoT.dz_remapped.compute()
    )
    # isoT["dTdz"] = -1 * isoT.Tmean.diff("pden_bins") / isoT.dz_remapped
    if debug:
        edgeT.cf.plot.line(marker="x", hue=bin_var, add_legend=False)
        # plt.gca().legend()

    isoT.coords[f"{bin_var}_bounds"] = cfxr.vertices_to_bounds(
        bins, out_dims=("bounds", bin_var)
    )
    isoT[bin_var].attrs["bounds"] = f"{bin_var}_bounds"
    isoT[bin_var].attrs.update(grad[bin_var].attrs)

    return isoT, edgeT
