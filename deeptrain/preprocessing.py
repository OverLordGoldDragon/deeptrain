# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd

from pathlib import Path
from .util._backend import WARN, NOTE

try:
    import lz4framed as lz4f
except:
    lz4f = None


def data_to_hdf5(savepath, batch_size, loaddir=None, data=None,
                 shuffle=False, compression='lzf', dtype=None,
                 load_fn=None, oversample_remainder=True, batches_dim0=False,
                 overwrite=None, verbose=1):
    """Convert data to hdf5-group (.h5) format, in `batch_size` sample sets.

    Arguments:
        savepath: str
            Absolute path to where to save file.
        batch_size: int
            Number of samples (dim0 slices) to save per file.
        loaddir: str
            Absolute path to directory from which to load data.
        data: np.ndarray / list[np.ndarray]
            Shape: `(samples, *)` or `(batches, samples, *)` (must use
            `batches_dim0=True`. With former, if `len(data) == 320` and
            `batch_size == 32`, will make a 10-set .h5 file.
        shuffle: bool
            Whether to shuffle samples (dim0 slices).
        compression: str
            Compression type to use. kwarg to `h5py.File().create_dataset()`.
        dtype: str / np.dtype
            Savefile dtype; kwarg to `.create_dataset()`.
            Defaults to data's dtype.
        load_fn: function / callable
            Used on supported paths (.npy) in `loaddir` to load data.
        oversample_remainder: bool. Relevant only when passing `data`.
            - True -> randomly draw (remainer - batch_size) samples to fill
              incomplete batch.
            - False -> drop remainder samples.
        batches_dim0: bool
            Assume shapes - True: `(batches, samples, *)`; False: `(samples, *)`.
        overwrite: bool / None
            If `savepath` file exists,

                - True  -> replace it
                - False -> don't replace it
                - None  -> ask confirmation via user input

        verbose: bool
            Whether to print preprocessing progress.

    **Notes**:

    - If supplying `loaddir` instead of `data`, will iteratively load files
      with supported format (.npy). `len()` of loaded file must be an integer
      fraction multiple of `batch_size`, <= 1. So `batch_size == 32` and
      `len() == 16` works, but `len() == 48` or `len() == 24` doesn't.
    """
    def _validate_args(savepath, loaddir, data, load_fn):
        def _validate_extensions(loaddir):
            supported = ('.npy',)
            extensions = list(set(x.suffix for x in Path(loaddir).iterdir()))
            if len(extensions) > 1:
                raise ValueError("cannot have more than one file extensions in "
                                 "`loaddir`; found %s" % ', '.join(extensions))
            elif load_fn is None and extensions[0] not in supported:
                raise ValueError(("unsupported file extension {}; supported "
                                  "are: {}. Alternatively, pass in `load_fn` "
                                  "that takes paths & index as arguments"
                                  ).format(extensions[0], ', '.join(supported)))

        if loaddir is None and data is None:
            raise ValueError("one of `loaddir` or `data` must be not None")
        if loaddir is not None and data is not None:
            raise ValueError("can't use both `loaddir` and `data`")
        if data is not None and load_fn is not None:
            print(WARN, "`load_fn` ignored with `data != None`")

        if Path(savepath).suffix != '.h5':
            print(WARN, "`savepath` extension must be '.h5'; will append")
            savepath += '.h5'
        _validate_savepath(savepath, overwrite)

        if loaddir is not None:
            _validate_extensions(loaddir)

        return savepath

    def _process_remainder(remainder, data, oversample_remainder, batch_size):
        action = "will" if oversample_remainder else "will not"
        print(("{} remainder samples for `batch_size={}`; {} oversample"
               ).format(int(remainder), batch_size, action))

        if oversample_remainder:
            to_oversample = batch_size - remainder
            idxs = np.random.randint(0, len(data), to_oversample)
            data = np.vstack([data, data[idxs]])
        else:
            data = data[:-remainder]
        return data

    def _get_data_source(loaddir, data, batch_size, compression, shuffle):
        source = (data if data is not None else
                  [str(x) for x in Path(loaddir).iterdir() if not x.is_dir()])
        if shuffle:
            np.random.shuffle(source)

        if verbose:
            comp = compression if compression is not None else "no"
            shuf = "with" if shuffle else "without"
            print(("Making {}-size batches from {} extractables, using {} "
                   "compression, {} shuffling").format(
                       batch_size, len(source), comp, shuf))
        return source

    def _make_batch(source, j, batch_size, load_fn, verbose):
        def _get_data(source, j, load_fn):
            def _load_data(source, j, load_fn):
                if load_fn is not None:
                    return load_fn(source, j)
                path = source[j]
                if Path(path).suffix == '.npy':
                    return np.load(path)
            try:
                return _load_data(source, j, load_fn)
            except:
                return source[j]
        X = []
        while sum(map(len, X)) < batch_size:
            if j == len(source):
                print(WARN, "insufficient samples in extractable to make "
                      "batch; terminating")
                return None, j
            X.append(_get_data(source, j, load_fn))
            j += 1
            if sum(map(len, X)) > batch_size:
                raise ValueError("`batch_size` exceeded; {} > {}".format(
                    sum(map(len, X)), batch_size))
            if verbose:
                print(end='.')
        return np.vstack(X), j

    def _make_hdf5(hdf5_file, source, batch_size, dtype, load_fn, verbose):
        j, set_num = 0, 0
        while j < len(source):
            batch, j = _make_batch(source, j, batch_size, load_fn, verbose)
            if batch is None:
                break
            dtype = dtype = batch.dtype
            hdf5_file.create_dataset(str(set_num), data=batch, dtype=dtype,
                                     chunks=True, compression=compression)
            if verbose:
                print('', set_num, 'done', flush=True)
            set_num += 1
        return set_num - 1

    savepath = _validate_args(savepath, loaddir, data, load_fn)

    if not batches_dim0 and data is not None:
        remainder = len(data) % batch_size
        if remainder != 0:
            data = _process_remainder(remainder, data, oversample_remainder,
                                      batch_size)
        n_batches = len(data) // batch_size
        data = data.reshape(n_batches, batch_size, *data.shape[1:])

    source = _get_data_source(loaddir, data, batch_size, compression, shuffle)

    with h5py.File(savepath, mode='w', libver='latest') as hdf5_file:
        last_set_num = _make_hdf5(hdf5_file, source, batch_size, dtype,
                                  load_fn, verbose)
    if verbose:
        print(last_set_num + 1, "batches converted & saved as .hdf5 to",
              savepath)


def numpy_to_lz4f(data, savepath=None, level=9, overwrite=None):
    """Do lz4-framed compression on `data`. (Install compressor via
    `!pip install py-lz4framed`)

    Arguments:
        data: np.ndarray
            Data to compress.
        savepath: str
            Path to where to save file.
        level: int
            1 to 9; higher = greater compression
        overwrite: bool
            If `savepath` file exists,

                - True  -> replace it
                - False -> don't replace it
                - None  -> ask confirmation via user input

    **Returns**:
        np.ndarray - compressed array.

    **Example**:

    >>> numpy_to_lz4f(savedata, savepath=path)
    ...
    >>> # load & decompress
    >>> bytes_npy = lz4f.decompress(np.load(path))
    >>> loaddata = np.frombuffer(bytes_npy,
    ...                          dtype=savedata.dtype,  # must be original's
    ...                          ).reshape(*savedata.shape)
    """

    if lz4f is None:
        raise Exception("cannot convert to lz4f without `lz4framed` installed; "
                        "run `pip install py-lz4framed`")
    data = data.tobytes()
    data = lz4f.compress(data, level=level)

    if savepath is not None:
        if Path(savepath).suffix != '.npy':
            print(WARN, "`savepath` extension must be '.npy'; will append")
            savepath += '.npy'
        _validate_savepath(savepath, overwrite)
        np.save(savepath, data)
        print("lz4f-compressed data saved to", savepath)
    return data


def numpy_data_to_numpy_sets(data, labels, savedir=None, batch_size=32,
                             shuffle=True, data_basename='batch',
                             oversample_remainder=True, overwrite=None,
                             verbose=1):
    """Save `data` in `batch_size` chunks, possibly shuffling samples.

    Arguments:
        data: np.ndarray
            Data to batch along `labels` & save.
        labels: np.ndarray
            Labels to batch along `data` and save.
        savedir: str / None
            Directory in which to save processed data. If None, won't save.
        batch_size: int
            Number of samples (dim0 slices) to form a 'set' with
        data_basename: str
            Will save with this prepending set numbering - e.g.: 'batch__1.npy',
            'batch__2.npy' ...
        oversample_remainder: bool
            - True -> randomly draw (remainer - batch_size) samples to fill
              incomplete batch.
            - False -> drop remainder samples.
        overwrite: bool / None
            If `savepath` file exists,

                - True  -> replace it
                - False -> don't replace it
                - None  -> ask confirmation via user input

        verbose: bool
            Whether to print preprocessing progress.

    Returns:
        data, labels: processed `data` & `labels`.
    """
    def _process_remainder(remainder, data, labels, oversample_remainder,
                           batch_size):
        action = "will" if oversample_remainder else "will not"
        print(("{} remainder samples for `batch_size={}`; {} oversample"
               ).format(int(remainder), batch_size, action))

        if oversample_remainder:
            to_oversample = batch_size - remainder
            idxs = np.random.randint(0, len(data), to_oversample)
            data = np.vstack([data, data[idxs]])
            labels = np.vstack([labels, labels[idxs]])
        else:
            data = data[:-remainder]
            labels = labels[:-remainder]
        return data, labels

    def _save(data, labels, savedir, verbose):
        labels_path = os.path.join(savedir, "labels.h5")
        labels_hdf5 = h5py.File(labels_path, mode='w', libver='latest')

        for set_num, (x, y) in enumerate(zip(data, labels)):
            set_num = str(set_num + 1)
            name = "{}__{}.npy".format(data_basename, set_num)
            savepath = os.path.join(savedir, name)

            _validate_savepath(savepath, overwrite)
            np.save(savepath, x)

            labels_hdf5.create_dataset(set_num, data=y, dtype=data.dtype)
            if verbose:
                print("[{}/{}] {}-sample batch {} processed & saved".format(
                    set_num, len(data), batch_size, name))

        labels_hdf5.close()
        if verbose:
            print("{} label sets saved to {}".format(len(data), labels_path))

    if labels.ndim == 1:
        labels = np.expand_dims(labels, 1)

    remainder = len(data) % batch_size
    if remainder != 0:
        data, labels = _process_remainder(remainder, data, labels,
                                          oversample_remainder, batch_size)
    if shuffle:
        idxs = np.arange(0, len(data))
        np.random.shuffle(idxs)
        data, labels = data[idxs], labels[idxs]
        print("`data` & `labels` samples shuffled")

    n_batches = len(data) / batch_size
    if not n_batches.is_integer():
        raise Exception(("len(data) must be divisible by "
                         "`batch_size` ({} / {} = {})").format(
                             len(data), batch_size, n_batches))
    data = data.reshape(int(n_batches), batch_size, *data.shape[1:])
    labels = labels.reshape(int(n_batches), batch_size, *labels.shape[1:])

    if savedir is not None:
        _save(data, labels, savedir, verbose)
    return data, labels


def numpy2D_to_csv(data, savepath=None, batch_size=None, columns=None,
                   sample_dim=1, overwrite=None):
    """Save 2D data as .csv.

    Arguments:
        data: np.ndarray
            Data to save, shaped `(batches, samples)`.
        savepath: str
            Path to where save to file.
        batch_size: int
            Number of rows per column; can differ from `data`'s.
        columns: list of str
            Column names for the data frame; defaults to enumerate columns
            (0, 1, 2, ...)
        sample_dim: int
            Dimension applicable to `batch_size`.
        overwrite: bool
            If `savepath` file exists,

                - True  -> replace it
                - False -> don't replace it
                - None  -> ask confirmation via user input

    **Example**:

    Suppose we have `labels`, 16 per batch, and 8 batches. Each batch is shaped
    (16,) - stacked, (8, 16). Dim 1 is thus `sample_dim`. Also see
    examples/preprocessing/timeseries.py.

    >>> data.shape == (8, 16)  # (num_batches, samples)
    >>> numpy2D_to_csv(data, "data.csv", batch_size=32, batch_dim=1)
    >>> # if it was (samples, num_batches), sample_dim would be 0.
    ... # This will make a DataFrame of 4 columns, 32 rows per column,
    ... # reshaping `data` to correctly concatenate samples from dim0 to dim1.

    """
    def _process_data(data, batch_size, sample_dim):
        if data.ndim != 2:
            raise ValueError("`data` must be 2D (got data.ndim=%s)" % data.ndim)

        batch_size = batch_size or data.shape[1]
        if data.shape[1] != batch_size:
            try:
                # need to 'stack' samples dims, matching format of `data_to_hdf5`
                if sample_dim == 1:
                    data = data.reshape(-1, batch_size, order='C').T
                else:
                    # we seek to concatenate samples from batches contiguously;
                    # order='C' won't accomplish this if samples are in dim0
                    data = data.reshape(batch_size, -1, order='F')
            except Exception as e:
                raise Exception("could not reshape `data`; specify different "
                                "`batch_size`.\nErrmsg: %s" % e)
        return data

    data = _process_data(data, batch_size, sample_dim)
    if columns is None:
        columns = list(map(str, range(data.shape[1])))

    df = pd.DataFrame(data, columns=columns)

    if savepath is not None:
        _validate_savepath(savepath, overwrite)
        df.to_csv(savepath, index=False)
        print(len(df.columns), "batch labels saved to", savepath)
    return df


def _validate_savepath(savepath, overwrite):
    if not Path(savepath).is_file():
        return

    if overwrite is None:
        response = input(("Found existing file in `savepath`; "
                          "overwrite?' [y/n]\n"))
        if response == 'y':
            os.remove(savepath)
        else:
            raise SystemExit("program terminated.")
    elif overwrite is True:
        os.remove(savepath)
        print(NOTE, "removed existing file from `savepath`")
    else:
        raise SystemExit(("program terminated. (existing file in "
                          "`savepath` and `overwrite=False`)"))
