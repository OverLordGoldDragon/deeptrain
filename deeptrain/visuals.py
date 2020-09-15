# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from see_rnn import get_weights, get_outputs, get_gradients
from see_rnn import features_hist
from . import scalefig


def binary_preds_per_iteration(_labels_cache, _preds_cache, w=1, h=1):
    """Plots binary preds vs. labels in a heatmap, separated by batches,
    grouped by slices.

    To be used with sigmoid outputs (1 unit). Both inputs are to be shaped
    `(batches, slices, samples, *)`.

    Arguments:
        _labels_cache: list[np.ndarray]
            List of labels cached during training/validation; insertion order
            must match that of `_preds_cache` (i.e., first array should
            correspond to labels of same batch as predictions in `_preds_cache`).
        _preds_cache: list[np.ndarray]
            List of predictions cached during training/validation; see docs
            on `_labels_cache`.
        w, h: float
            Scale figure width & height, respectively.
    """
    N = len(_labels_cache)
    lc = np.asarray(_labels_cache)
    pc = np.asarray(_preds_cache)
    if lc.shape[-1] == 1:
        lc = lc.squeeze(axis=-1)
    if pc.shape[-1] == 1:
        pc = pc.squeeze(axis=-1)
    N, n, *_ = lc.shape

    fig, axes = plt.subplots(N, 1, figsize=(12 * w, N * (n / 6) ** .5))
    for ax in axes:
        ax.set_axis_off()

    # rows = batches
    # arr_rows = rows of plotted arrays for each subplot
    # arr_cols = columns of plotted arrays for each subplot
    for l, p, ax in zip(lc, pc, axes.flat):
        # show labels on top w/ double height for emphasis
        l = np.vstack([l[:1], l[:1]])
        # plot [labels, preds] stacked, (arr_rows, arr_cols) = (slices, samples)
        lp = np.vstack([l, p])

        ax.imshow(lp, cmap='bwr')
        ax.set_axis_off()
        ax.axis('tight')

    plt.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=.3)
    scalefig(fig)
    plt.show()


def binary_preds_distribution(_labels_cache, _preds_cache, pred_th, w=1, h=1):
    """Plots binary preds in a scatter plot, labeling dots according to
    their labels, and showing `pred_th` as a vertical line. Positive class (1)
    is labeled red, negative (0) blue; a red dot far left (close to 0) is
    hence a strongly misclassified positive class, and vice versa.

    To be used with sigmoid outputs (1 unit).

    Arguments:
        _labels_cache: list[np.ndarray]
            List of labels cached during training/validation; insertion order
            must match that of `_preds_cache` (i.e., first array should
            correspond to labels of same batch as predictions in `_preds_cache`).
        _preds_cache: list[np.ndarray]
            List of predictions cached during training/validation; see docs
            on `_labels_cache`.
        pred_th: float
            Predict threshold (e.g. 0.5), plotted as a vertical line.
        w, h: float
            Scale figure width & height, respectively.
    """
    def _get_pred_colors(labels_f):
        labels_f = np.expand_dims(labels_f, -1)
        N = len(labels_f)
        red  = np.array([1, 0, 0] * N).reshape(N, 3)
        blue = np.array([0, 0, 1] * N).reshape(N, 3)
        return labels_f * red + (1 - labels_f) * blue

    def _make_alignment_array(labels_f):
        N = len(labels_f)
        # estimate appropriate number of rows for the dots
        if N > 5000:
            n_rows = 24
        elif N > 3000:
            n_rows = 16
        else:
            n_rows = 10

        k = N / n_rows
        # decrement until is integer
        while not k.is_integer():
            n_rows -= 1
            k = N / n_rows
        return np.array(list(range(n_rows)) * int(k))

    def _plot(preds_f, pred_th, alignment_arr, colors):
        height = 4 if len(preds_f) < 5000 else 6
        fig, ax = plt.subplots(1, 1, figsize=(13 * w, height * h))
        ax.axvline(pred_th, color='black', linewidth=4)
        ax.scatter(preds_f, alignment_arr, c=colors)
        ax.set_yticks([])
        ax.set_xlim(-.02, 1.02)

        scalefig(fig)
        plt.show()

    preds_flat = np.asarray(_preds_cache).ravel()
    labels_flat = np.asarray(_labels_cache).ravel()
    colors = _get_pred_colors(labels_flat)
    alignment_arr = _make_alignment_array(labels_flat)

    _plot(preds_flat, pred_th, alignment_arr, colors)


def infer_train_hist(model, input_data, layer=None, keep_borders=True,
                     bins=100, xlims=None, fontsize=14, vline=None,
                     w=1, h=1):
    """Histograms of flattened layer output values in inference and train mode.
    Useful for comparing effect of layers that behave differently in train vs
    infrence modes (Dropout, BatchNormalization, etc) on model prediction (or
    intermediate activations).

    Arguments:
        model: models.Model / models.Sequential (keras / tf.keras)
            The model.
        input_data: np.ndarray / list[np.ndarray]
            Data to feed to `model` to fetch outputs. List of arrays for
            multi-input networks.
        layer: layers.Layer / None
            Layer whose outputs to fetch; defaults to last layer (output).
        keep_borders: bool
            Whether to keep the plots' bounding box.
        bins: int
            Number of histogram bins; kwarg to `plt.hist()`
        xlims: tuple[float, float] / None
            Histogram x limits. Defaults to min/max of flattened data per plot.
        fontsize: int
            Title font size.
        vline: float / None
            x-coordinate of vertical line to draw (e.g. predict threshold).
        w, h: float
            Scale figure width & height, respectively.
    """
    layer = layer or model.layers[-1]
    outs = [get_outputs(model, '', input_data, layer, learning_phase=0),
            get_outputs(model, '', input_data, layer, learning_phase=1)]
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True,
                             figsize=(13 * w, 6 * h))

    for i, (ax, out) in enumerate(zip(axes.flat, outs)):
        out = np.asarray(out).ravel()
        ax.hist(out, bins=bins)

        mode = "ON" if i == 0 else "OFF"
        ax.set_title("Train mode " + mode, weight='bold', fontsize=fontsize)
        if not keep_borders:
            ax.box(on=None)
        if vline is not None:
            ax.axvline(vline, color='r', linewidth=2)
        _xlims = xlims or (out.min(), out.max())
        ax.set_xlim(*_xlims)

    scalefig(fig)
    plt.show()


def layer_hists(model, _id='*', mode='weights', input_data=None, labels=None,
                omit_names='bias', share_xy=(0, 0), configs=None, **kw):
    """Histogram grid of layer weights, outputs, or gradients.

    Arguments:
        model: models.Model / models.Sequential (keras / tf.keras)
            The model.
        _id: str / int / list[str/int].
            - int -> idx; str -> name
            - idx: int. Index of layer to fetch, via model.layers[idx].
            - name: str. Name of layer (full or substring) to be fetched.
              Returns earliest match if multiple found.
            - list[str/int] -> treat each str element as name, int as idx.
              Ex: `['gru', 2]` gets (e.g.) weights of first layer with name
              substring 'gru', then of layer w/ idx 2.
            - `'*'` (wildcard) -> get (e.g.) outputs of all layers (except input)
              with 'output' attribute.
        mode: str in ('weights', 'outputs', 'gradients:weights',\
        'gradients:outputs')
            Whether to fetch layer weights, outputs, or gradients (w.r.t.
            outputs or weights).
        input_data: np.ndarray / list[np.ndarray] / None
            Data to feed to `model` to fetch outputs / gradients. List of arrays
            for multi-input networks. Ignored for `mode='weights'`.
        labels: np.ndarray / list[np.ndarray]
            Labels to feed to `model` to fetch gradients. List of arrays
            for multi-output networks.
            Ignored for `mode in ('weights', 'outputs')`.
        omit_names: str / list[str] / tuple[str]
            Names of weights to omit for `_id` specifying layer names. E.g.
            for `Dense`, `omit_names='bias'` will fetch only kernel weights.
            Ignored for `mode != 'weights'`.
        share_xy: tuple[bool, bool] / tuple[str, str]
            Whether to share x or y limits in histogram grid, respectively.
            kwarg to `plt.subplots()`; can be `'col'` or `'row'` for sharing
            along rows or columns, respectively.
        configs: dict / None
            kwargs to customize various plot schemes:

            - `'plot'`: passed partly to `ax.hist()` in `see_rnn.hist_clipped()`;
              include `peaks_to_clip` to adjust ylims with a
              number of peaks disregarded. See `help(see_rnn.hist_clipped)`.
              ax = subplots axis
            - `'subplot'`: passed to `plt.subplots()`
            - `'title'`: passed to `fig.suptitle()`; fig = subplots figure
            - `'tight'`: passed to `fig.subplots_adjust()`
            - `'annot'`: passed to `ax.annotate()`
            - `'save'`: passed to `fig.savefig()` if `savepath` is not None.
        kw: dict / kwargs
            kwargs passed to `see_rnn.features_hist`.
    """
    def _process_configs(configs, mode):
        defaults = {}
        if 'gradients' in mode or mode == 'outputs':
            defaults.update({'plot': dict(peaks_to_clip=2, annot_kw=None),
                             'subplot': dict(sharex=False, sharey=False)})
        if not configs:
            return defaults
        for name, _dict in defaults.items():
            if name not in configs:
                configs[name] = _dict
            else:
                for k, v in _dict.items():
                    if k not in configs[name]:
                        configs[name][k] = v
        return configs

    def _prevalidate(mode, input_data, labels):
        supported = ('weights', 'outputs', 'gradients',
                     'gradients:outputs', 'gradients:weights')
        if mode not in supported:
            raise ValueError(("unsupported `mode`: '{}'; supported are: {}"
                             ).format(mode, ', '.join(supported)))
        if 'gradients' in mode and (input_data is None or labels is None):
            raise ValueError("`input_data` or `labels` cannot be None for "
                             "'gradients'-based `mode`")
        if mode == 'outputs' and input_data is None:
            raise ValueError("`input_data` cannot be None for `mode='outputs'`")

    def _get_data(model, _id, mode, input_data, labels, omit_names, kw):
        if mode == 'weights':
            data = get_weights(model, _id, omit_names, as_dict=True)
        elif 'gradients' in mode:
            if mode in ('gradients', 'gradients:outputs'):
                data = get_gradients(model, _id, input_data, labels,
                                     mode='outputs', as_dict=True)
            else:
                data = get_gradients(model, _id, input_data, labels,
                                     mode='weights', as_dict=True)
        elif mode == 'outputs':
            data = get_outputs(model, _id, input_data, as_dict=True)

        data_flat = [x.ravel() for x in data.values()]
        return data_flat, list(data)

    configs = _process_configs(configs, mode)
    _prevalidate(mode, input_data, labels)
    data_flat, data_names = _get_data(model, _id, mode, input_data, labels,
                                      omit_names, kw)
    features_hist(data_flat, annotations=data_names, configs=configs,
                  share_xy=share_xy, **kw)


def viz_roc_auc(y_true, y_pred):
    """Plots the Receiver Operator Characteristic curve."""
    def _compute_roc_auc(y_true, y_pred):
        i_x = [(i, x) for (i, x) in enumerate(y_pred)]
        # order preds descending
        i_xs = list(sorted(i_x, key=lambda x: x[1], reverse=True))
        idxs = [d[0] for d in i_xs]
        # get labels at positions of descending predictions
        ys = y_true[idxs]

        p_inc = 1 / ys.sum()              # 1-class increment
        n_inc = 1 / (len(ys) - ys.sum())  # 0-class increment
        # points array to fill, shaped [num_points, (x, y)]
        pts = np.zeros((len(ys) + 1, 2))

        # fill (x_i, y_i) for i = 0, ..., num_points - 1
        for i, y in enumerate(ys):
            if y == 1:
                # x_i = x_{i-1}
                # y_i = y_{i-1} + p_inc
                pts[i + 1] = [pts[i][0], pts[i][1] + p_inc]
            else:
                # x_i = x_{i-1} + n_inc
                # y_i = y_{i-1}
                pts[i + 1] = [pts[i][0] + n_inc, pts[i][1]]

        score = np.trapz(pts[:, 1], pts[:, 0])  # trapezoid rule integral
        return pts, score

    def _plot(pts, score):
        kw = dict(fontsize=12, weight='bold')
        plt.scatter(pts[:, 0], pts[:, 1])

        plt.title("Receiver Operating Characteristic (AUC = %.3f)" % score, **kw)
        plt.xlabel("1 - specificity", **kw)
        plt.ylabel("sensitivity", **kw)
        plt.gcf().set_size_inches(6, 6)

        scalefig(plt.gcf())
        plt.show()

    # standardize
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    pts, score = _compute_roc_auc(y_true, y_pred)
    _plot(pts, score)


def get_history_fig(self, plot_configs=None, w=1, h=1):
    """Plots train / validation history according to `plot_configs`.

    Arguments:
        plot_configs: dict / None
            See :data:`_DEFAULT_PLOT_CFG`. If None, defaults to
            `TrainGenerator.plot_configs` (which itself defaults to `_PLOT_CFG`
            in `configs.py`).
        w, h: float
            Scale figure width & height, respectively.

    `plot_configs` is structured as follows:

    >>> {'fig_kw': fig_kw,
    ...  '0': {reserved_name: value,
    ...        plt_kw: value},
    ...  '1': {reserved_name: value,
    ...        plt_kw: value},
    ...  ...}

    - `fig_kw`: dict, passed to `plt.subplots(**fig_kw)`
    - `reserved_name`: str, one of `('metrics', 'x_ticks', 'vhlines',
      'mark_best_cfg', 'ylims', 'legend_kw')`. Used to configure supported custom
      plot behavior (see "Builtin plot customs" below).
    - `plt_kw`: str, name of kwarg to pass directly to `plt.plot()`.
    - `value`: depends on key; see default `plot_configs` in
      :data:`_DEFAULT_PLOT_CFG` and :meth:`misc._make_plot_configs_from_metrics`.

    Only `'metrics'` and `'x_ticks'` keys are required for each dict - others
    have default values.

    **Builtin plot customs**: (`reserved_name`)

    - `'metrics'` (required): names of metrics to plot from histories, as
      `{'train': train_metrics, 'val': val_metrics}` (at least one metric name
      required, for only one of train/val - need to have "something" to plot).
    - `x_ticks'` (required): x-coordinates of respective metrics, of same `len()`.
    - `'vhlines'`: dict['v' / 'h': float]. vertical/horizontal lines; e.g.
      `{'v': 10}` will draw a vertical line at x = 10, and `{'h': .5}` at y = .5.
    - `'mark_best_cfg'`: `{'train': metric_name}` or `{'val': metric_name}` and
      (optional) `{'max_is_best: bool}` pairs. Will mark plot to indicate
      a metric optimum (max (if `'max_is_best'`, the default) or min).
    - `'ylims'`: y-limits of plot panes.
    - `'legend_kw'`: passed to `plt.legend()`; if None, no legend is drawn.

    **Defaults handling**:

    Keys and subkeys, where absent, will be filled from configs returned by
    :meth:`misc._make_plot_configs_from_metrics`.

        - If plot pane `'0'` is lacking entirely, it'll be copied from the
          defaults.
        - If subkey `'color'` in dict with key `'0'` is missing, will fill
          from `defaults['0']['color']`.

    **Further info**:

    - Every key's iterable value (list, etc) must be of same len as number of
      metrics in `'metrics'`; this is ensured within `cfg_fn`.
    - Metrics are plotted in order of insertion (at both dict and list level),
      so later metrics will carry over to additional plot panes if number of
      metrics exceeds `plot_first_pane_max_vals`; see `cfg_fn`.
    - A convenient option is to change `_PLOT_CFG` in `configs.py` and pass
      `plot_configs=None` to `TrainGenerator.__init__`; will internally call
      `cfg_fn`, which validates some configs and tries to fill what's missing.
    - Above, `cfg_fn` == :meth:`misc._make_plot_configs_from_metrics`
    """
    def _unpack_plot_kw(config):
        reserved_keys = ('metrics', 'x_ticks', 'vhlines',
                         'mark_best_cfg', 'ylims', 'legend_kw')
        metric_keys = list(config['metrics'])  # 'train', 'val'
        values_per_key = sum(len(config['metrics'][x]) for x in metric_keys)

        plot_kw = []
        for i in range(values_per_key):
            plot_kw.append({key: config[key][i] for key in config
                            if key not in reserved_keys})
        return plot_kw

    def _equalize_ticks_range(x_ticks, metrics):
        max_value = max(np.max(ticks) for ticks in x_ticks if len(ticks) > 0)

        if not all(ticks[-1] == max_value for ticks in x_ticks):
            raise Exception(("last xtick isn't greatest (got {}, max is {})"
                             ).format(", ".join(str(t[-1]) for t in x_ticks),
                                      max_value))
        if not all(len(t) == len(m) for t, m in zip(x_ticks, metrics.values())):
            raise Exception(("len of ticks doesn't match len of metrics:\n{}"
                             ).format("\n".join("%s %s %s" % (len(t), len(m), n)
                                                for t, (n, m) in
                                                zip(x_ticks, metrics.items()))))
        return x_ticks

    def _equalize_metric_names(config):
        metrics_cfg = config['metrics']

        if 'train' in metrics_cfg:
            for idx, name in enumerate(metrics_cfg['train']):
                metrics_cfg['train'][idx] = self._alias_to_metric_name(name)
        if 'val'   in metrics_cfg:
            for idx, name in enumerate(metrics_cfg['val']):
                metrics_cfg['val'][idx]   = self._alias_to_metric_name(name)

    def _unpack_vhlines(config):
        vhlines = {'v': [], 'h': []}
        for vh in vhlines:
            vhline = config['vhlines'][vh]
            if isinstance(vhline, (float, int, list, tuple, np.ndarray)):
                vhlines[vh] = vhline
            elif vhline == '_val_hist_vlines':
                vhlines[vh] = self._val_hist_vlines or None
            elif vhline == '_hist_vlines':
                vhlines[vh] = self._hist_vlines or None
            else:
                raise ValueError("unsupported `vhlines` in `plot_configs`:",
                                 vhline)
        return vhlines

    def _unpack_ticks_and_metrics(config):
        def _get_mark_best_idx(metrics, name, mark_best_cfg, val):
            expects_val = bool('val' in mark_best_cfg)
            expected_name = list(mark_best_cfg.values())[0]

            if not val and expects_val:
                return
            elif val and expects_val         and name == expected_name:
                return len(metrics) - 1
            elif not val and not expects_val and name == expected_name:
                return len(metrics) - 1

        x_ticks, metrics = [], {}
        mark_best_cfg = config.get('mark_best_cfg', None)
        mark_best_idx = None
        # TODO replace mark_best_idx w/ name?
        if 'train' in config['metrics']:
            for i, name in enumerate(config['metrics']['train']):
                metrics[f'train:{name}'] = self.history[name]
                x_ticks.append(getattr(self, config['x_ticks']['train'][i]))
                if mark_best_cfg is not None and mark_best_idx is None:
                    mark_best_idx = _get_mark_best_idx(metrics, name,
                                                       mark_best_cfg, val=False)
        if 'val' in config['metrics']:
            for i, name in enumerate(config['metrics']['val']):
                metrics[f'val:{name}'] = self.val_history[name]
                x_ticks.append(getattr(self, config['x_ticks']['val'][i]))
                if mark_best_cfg is not None and mark_best_idx is None:
                    mark_best_idx = _get_mark_best_idx(metrics, name,
                                                       mark_best_cfg, val=True)
        return x_ticks, metrics, mark_best_idx

    if plot_configs is None:
        plot_configs = self.plot_configs

    subplot_configs = {k: v for k, v in plot_configs.items() if k != 'fig_kw'}
    if not all(('metrics' in cfg and 'x_ticks' in cfg)
               for cfg in subplot_configs.values()):
        raise ValueError("all dicts in `plot_configs` (except w/ 'fig_kw' key) "
                         "must include 'metrics', 'x_ticks'")

    fig, axes = plt.subplots(len(subplot_configs), 1, **plot_configs['fig_kw'])
    axes = np.atleast_1d(axes)

    for config, axis in zip(subplot_configs.values(), axes):
        _equalize_metric_names(config)
        x_ticks, metrics, mark_best_idx = _unpack_ticks_and_metrics(config)
        x_ticks = _equalize_ticks_range(x_ticks, metrics)

        plot_kw = _unpack_plot_kw(config)
        if config.get('vhlines', None) is not None:
            vhlines  = _unpack_vhlines(config)
        else:
            vhlines = None
        ylims = config.get('ylims', (0, 2))
        legend_kw = config.get('legend_kw', None)

        max_is_best = config.get('mark_best_cfg', {}).get('max_is_best', True)
        _plot_metrics(x_ticks, metrics, plot_kw, mark_best_idx, max_is_best,
                      axis=axis, vhlines=vhlines, ylims=ylims,
                      legend_kw=legend_kw, key_metric=self.key_metric,
                      metric_name_to_alias_fn=self._metric_name_to_alias)

    subplot_scaler = .5 * len(axes)
    fig.set_size_inches(14 * w, 11 * h * subplot_scaler)
    scalefig(fig)
    plt.close(fig)
    return fig


def _plot_metrics(x_ticks, metrics, plot_kw, mark_best_idx=None,
                  max_is_best=True, axis=None, vhlines=None,
                  ylims=(0, 2), legend_kw=None, key_metric='loss',
                  metric_name_to_alias_fn=None):
    """Plots metrics according to inputs passed by :meth:`get_history_fig`."""
    def _plot_vhlines(vhlines, ax):
        def non_iterable(x):
            return not isinstance(x, (list, tuple, np.ndarray))
        vlines, hlines = vhlines.get('v', None), vhlines.get('h', None)

        if vlines is not None:
            if non_iterable(vlines):
                vlines = [vlines]
            [ax.axvline(l, color='k', linewidth=2) for l in vlines if l]
        if hlines is not None:
            if non_iterable(hlines):
                hlines = [hlines]
            [ax.axhline(l, color='k', linewidth=2) for l in hlines if l]

    def _mark_best_metric(x_ticks, metrics, mark_best_idx, ax):
        metric = list(metrics.values())[mark_best_idx]

        best_fn = np.max if max_is_best else np.min
        x_best_idx = np.where(metric == best_fn(metric))[0][0]
        x_best = x_ticks[mark_best_idx][x_best_idx]

        ax.plot(x_best, best_fn(metric), 'o', color=[.3, .95, .3],
                markersize=15, markeredgewidth=4, markerfacecolor='none')

    def _make_legend_label(name, bold):
        mode, metric_name = name.split(':')
        if metric_name_to_alias_fn:
            metric_name = metric_name_to_alias_fn(metric_name)
        label = "{} ({})".format(metric_name, mode)
        if bold:
            label = f"$\\bf{label}$"
        return label

    def _plot_main(x_ticks, metrics, plot_kw, legend_kw, ax):
        bold = bool(legend_kw.get('weight', None) == 'bold')
        for ticks, name, kws in zip(x_ticks, metrics, plot_kw):
            if legend_kw is not None:
                kws['label'] = _make_legend_label(name, bold)
            ax.plot(ticks, metrics[name], **kws)

    vhlines = vhlines or {'v': None, 'h': None}
    ax = axis if axis else plt.subplots()[1]
    _plot_main(x_ticks, metrics, plot_kw, legend_kw, ax)

    if legend_kw is not None:
        legend_kw = legend_kw.copy()  # ensure external dict unaffected
        legend_kw.pop('weight', None)  # invalid kwarg (but used above)
        ax.legend(**legend_kw)
    if vhlines is not None:
        _plot_vhlines(vhlines, ax)

    if mark_best_idx is not None:
        _mark_best_metric(x_ticks, metrics, mark_best_idx, ax)

    xmin = min(np.min(ticks) for ticks in x_ticks)
    xmax = max(np.max(ticks) for ticks in x_ticks)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(*ylims)
