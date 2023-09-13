import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import copy
# from candas.learn import GP, GLM

from candas.learn import ParameterArray, UncertainParameterArray, ParameterSet, Standardizer
from candas.learn import parray, uparray


def plot_uparray_ci(x_pa, y_upa, ci=0.95, ax=None, **kwargs):
    ax = plt.gca() if ax is None else ax

    l = y_upa.dist.ppf((1 - ci) / 2)
    u = y_upa.dist.ppf((1 + ci) / 2)

    ax.fill_between(x_pa.values(), l, u, **kwargs)
    return ax


def plot_uparray_mean(x_pa, y_upa, ax=None, **kwargs):
    ax = plt.gca() if ax is None else ax

    ax.plot(x_pa.values(), y_upa.μ, **kwargs)
    return ax


def plot_uparray(x_pa, y_upa, ci=0.95, ax=None, palette=None, line_kws=None, fill_kws=None,
                 x_scale='standardized', y_scale='natural'):
    ax = plt.gca() if ax is None else ax

    line_kws = dict() if line_kws is None else line_kws
    fill_kws = dict() if fill_kws is None else fill_kws
    palette = sns.cubehelix_palette() if palette is None else palette
    palette = sns.color_palette(palette) if type(palette) is str else palette

    line_defaults = dict(lw=2, color=palette[-2], zorder=0)
    fill_defaults = dict(lw=2, facecolor=palette[1], zorder=-1, alpha=0.5)

    if x_scale == 'standardized':
        x = x_pa.z
    elif x_scale == 'transformed':
        x = x_pa.t
    else:
        x = x_pa

    if y_scale == 'standardized':
        y = y_upa.z
    elif y_scale == 'transformed':
        y = y_upa.t
    else:
        y = y_upa

    plot_uparray_mean(x, y, ax=ax, **{**line_defaults, **line_kws})
    plot_uparray_ci(x, y, ci=ci, ax=ax, **{**fill_defaults, **fill_kws})

    ax.set_ylabel(y_upa.name)
    ax.set_xlabel(x_pa.names[0])

    for (axis, scale, array) in zip(['x', 'y'], [x_scale, y_scale], [x_pa, y_upa]):
        if scale == 'standardized':
            unstandardize_axis_labels(ax, axis, array)
    return ax


def contourf_uparray(x_pa, y_pa, z_upa, ax=None, x_scale='standardized', y_scale='standardized', z_scale='natural', uncertainty = False, vminvmax=None,
                     **kwargs):
    ax = plt.gca() if ax is None else ax

    cmap_crest = sns.color_palette("crest", as_cmap=True)

    cmap_flare = sns.color_palette("flare", as_cmap=True)

    if x_scale == 'standardized':
        x = x_pa.z
    elif x_scale == 'transformed':
        x = x_pa.t
    else:
        x = x_pa

    if y_scale == 'standardized':
        y = y_pa.z
    elif y_scale == 'transformed':
        y = y_pa.t
    else:
        y = y_pa

    if z_scale == 'standardized':
        z = z_upa.z
    elif z_scale == 'transformed':
        z = z_upa.t
    else:
        z = z_upa

    if uncertainty:
        if vminvmax is None:
            levels = 16
        else:
            levels = np.linspace(vminvmax[0], vminvmax[1], 16)
        defaults = dict(levels=levels, cmap=cmap_crest)
        contour = ax.contourf(x, y, 2 * z.σ, **{**defaults, **kwargs})
    else:
        if vminvmax is None:
            levels = 16
        else:
            levels = np.linspace(vminvmax[0], vminvmax[1], 16)
        defaults = dict(levels=levels, cmap='pink')
        contour = ax.contourf(x, y, z.μ, **{**defaults, **kwargs})

    ax.set_ylabel(y_pa.names[0])
    ax.set_xlabel(x_pa.names[0])

    for (axis, scale, array) in zip(['x', 'y'], [x_scale, y_scale], [x_pa, y_pa]):
        if scale == 'standardized':
            unstandardize_axis_labels(ax, axis, array)
    return contour


def unstandardize_axis_labels(ax, axis, array):
    if axis == 'x':
        ticks = ax.get_xticks()
        set_labels = ax.set_xticklabels
    elif axis == 'y':
        ticks = ax.get_yticks()
        set_labels = ax.set_yticklabels
    elif axis == 'z':
        ticks = ax.get_zticks()
        set_labels = ax.set_zticklabels

    if isinstance(array, ParameterArray):
        name = array.names[0]
        stdzr = array.stdzr
    elif isinstance(array, UncertainParameterArray):
        name = array.name
        stdzr = array.dstdzr.stdzr
    else:
        raise TypeError('Argument "array" must be either a ParameterArray or an UncertainParameterArray.')

    ticks = parray(**{name: ticks}, stdzr=stdzr, stdzd=True)
    set_labels(ticks.values())


class RegressionPlotter:
    # # TODO: improve `ticks` handling
    # if ticks is None:
    #     ticks = dict()
    # if not isinstance(ticks, dict):
    #     raise TypeError('"ticks" must be a dictionary')
    #
    # if 'GC' in self.dims:
    #     # limits.setdefault('GC', self.parray(**{'GC': [0.075, 0.925]}))
    #     ticks.setdefault('GC', self.parray(**{'GC': [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]}))
    # if 'BP' in self.dims:
    #     # limits.setdefault('BP', self.parray(**{'BP': [10, 800]}))
    #     ticks.setdefault('BP', self.parray(**{'BP': [15, 25, 50, 100, 200, 400, 800]}))
    #
    # self.ticks = ticks

    def __init__(self, av_limits=None, er_limits=None, av_label=None, er_label=None, ticks=None, cmap=None,
                 av_norm=None, av_vmin=None, av_vmax=None, er_norm=None, er_vmin=None, er_vmax=None):
        self.av_limits = av_limits
        self.er_limits = er_limits
        self.av_label = av_label
        self.er_label = 'Uncertainty\n(Standard Deviations)' if er_label is None else er_label

        default_ticks = {'GC': [15, 25, 50, 100, 200, 400, 800],
                         'BP': [10, 20, 40, 60, 80, 90]}
        self.ticks = default_ticks if ticks is None else {**default_ticks, **ticks}
        self.tick_precision = {}
        self.tick_format = {}
        self._default_precision = 3
        self._default_format = 'g'
        self.cmap = sns.color_palette("flare_r", as_cmap=True) if cmap is None else cmap
        self.av_norm = av_norm
        self.av_vmin = av_vmin
        self.av_vmax = av_vmax
        self.er_norm = er_norm
        self.er_vmin = er_vmin
        self.er_vmax = er_vmax

        # self.limits = {'μ': [0.4, 1.2], 'σ': [0.2, 1.8]}

    def plot_mean_contourf(self, Xdata, Xdims, Zdata, ax=None, n_levels=16, update=True, colorbar=True,
                           transformed_X=True, transformed_Z=True, **kwargs):

        assert isinstance(Xdata, ParameterArray)
        assert isinstance(Xdims, list) and len(Xdims) == 2
        assert isinstance(Zdata, UncertainParameterArray)

        ax = plt.subplots(1, 1, figsize=(10, 4))[1] if ax is None else ax

        stdzr = Xdata.stdzr
        xvar = Xdims[0]
        yvar = Xdims[1]
        X = copy(Xdata[xvar])
        Y = copy(Xdata[yvar])
        Z = copy(Zdata)

        zlim = [Z.μ.min(), Z.μ.max()] if self.av_limits is None else self.av_limits
        Z.μ = np.clip(Z.μ, *zlim)

        if update or self.av_limits is None:
            self.av_limits = zlim

        defaults = {
            'vmin': self.av_vmin,
            'vmax': self.av_vmax,
            'norm': self.av_norm,
            'cmap': self.cmap,
        } if not update else {}

        defaults['levels'] = np.linspace(*zlim, n_levels + 1)

        defaults['extend'] = {
            (True, True): 'both',
            (True, False): 'min',
            (False, True): 'max',
            (False, False): 'neither',
        }[(Z.μ.min() < zlim[0], Z.μ.max() > zlim[1])]

        kwargs = {**defaults, **kwargs}

        if transformed_X:
            X = X.t
            Y = Y.t
        if transformed_Z:
            Z = Z.t

        contours = ax.contourf(X, Y, Z.μ, **kwargs)
        plot_objects = {'contours': contours}

        xticks = self.ticks.get(xvar)
        yticks = self.ticks.get(yvar)
        ticks_lists = [xticks, yticks]
        getters = [ax.get_xticks, ax.get_yticks]
        setters = [ax.set_xticks, ax.set_yticks]
        namers = [ax.set_xticklabels, ax.set_yticklabels]

        for dim, ticks, get_ticks, set_ticks, set_ticklabels in zip(Xdims, ticks_lists, getters, setters, namers):
            if ticks is None:
                self.ticks[dim] = get_ticks()
                continue
            precision = self.tick_precision.get(dim, self._default_precision)
            format = self.tick_format.get(dim, self._default_format)
            names = [f'{t:.{precision}{format}}' for t in ticks]
            if transformed_X:
                ticks = stdzr.transform(dim, ticks, lg10_Copies=Xdata.get('lg10_Copies'))
            set_ticks(ticks)
            set_ticklabels(names)

        if colorbar:
            cbar = plt.colorbar(contours, ax=ax, pad=0.05, format='%.2g')
            plot_objects['colorbar'] = cbar
            cbar.set_label(self.av_label)
            zticks = self.ticks.get(Z.name)
            if zticks is None:
                zticks = cbar.get_ticks() if not transformed_Z else stdzr.untransform(Z.name, zticks)
                self.ticks[Z.name] = zticks
            else:
                precision = self.tick_precision.get(Z.name, self._default_precision)
                format = self.tick_format.get(Z.name, self._default_format)
                names = [f'{t:.{precision}{format}}' for t in ticks]
                if transformed_Z:
                    zticks = stdzr.transform(Z.name, zticks)
                cbar.set_ticks(zticks)
                cbar.set_ticklabels(names)

        if update or self.vmin is None:
            self.av_vmin = contours.vmin
        if update or self.vmax is None:
            self.av_vmax = contours.vmax
        if update or self.norm is None:
            self.av_norm = contours.norm
        if update or self.cmap is None:
            self.cmap = contours.cmap

        return plot_objects

    # clist = np.vstack([[bkg, bkg, bkg, alpha] for alpha in np.linspace(0, 0.6, 256)])
    # ercmap = mpl.colors.LinearSegmentedColormap.from_list('VSUP', clist, er_levels)
    #
    # sm = plt.cm.ScalarMappable(cmap=ercmap, norm=er_kws['norm'])
    # sm.set_clim(*er_clim)
    # sm.set_array(Z_err)
    #
    # er_pc = ax.contourf(X, Y, Z_err, cmap=ercmap, levels=np.linspace(*er_clim, er_levels + 1), **er_kws)
    #
    # if er_cbar:
    #     er_cbar = plt.colorbar(ax=ax, mappable=sm, pad=0.15,
    #                            boundaries=np.linspace(*er_clim, er_levels + 1), format='%.2g')
    #     er_cbar.set_label(er_label)
    #     if er_cticks is not None:
    #         er_cbar.set_ticks(er_cticks)
    #     if er_cticklabels is not None:
    #         er_cbar.set_ticklabels([f'{l:.2g}' for l in er_cticklabels])
