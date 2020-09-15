import os
import h5py
import numpy as np
import pandas as pd

from pathlib import Path
from types import LambdaType
from ._backend import lz4f, IMPORTS


class DataLoader():
    """Loads data from files to feed :class:`DataGenerator`. Builtin methods for
    handling various data & file formats. Is set within
    :meth:`DataGenerator._infer_and_set_info`.

    **Arguments**:
        path: str
            Path to directory or file from which to get load data. If file,
            or if directory contains one file that isn't of "opposite path"
            (`labels_path` if `path == data_path`, and vice versa), then uses
            Dataset mode of operation (see below) - else directory.
        loader: str / function / None
            Name of builtin function, or a custom function with input
            signature `(self, set_num)`. Loads data from directory, or
            dataset file if `_is_dataset`. If None, defaults to a builtin
            as determined by :meth:`load_fn` setter.
        dtype: str / dtype
            Dtype of data to load, required by some loaders (`numpy-lz4f`).
        batch_shape: tuple[int]
            Full batch shape of data to load, required by some loaders
            (`numpy-lz4f`).
        base_name: str
            Name common to all filenames in directory, used to delimit by
            `set_num` (e.g. `data1.npy`, `data2.npy`, etc)
        ext: str
            Extension of file(s) to load (e.g. `.npy`, `.h5`, etc).
        filepaths: list[str]
            Paths to files to load.

    **Builtin loaders**: see :data:`_BUILTINS`

    **Custom loaders**:

    Simplest option is to inherit `DataLoader` and override :meth:`_get_loader`
    to return the custom loader; :class:`DataGenerator` will handle the rest.
    Fully custom ones require:

        - `__init__` with same input signature as `DataLoader.__init__`.
        - `load_fn` method with `(self, set_num)` input signature, loading data
          from a directory / file
        - `_get_loader` method with `(self, loader)` input signature, returning
          the custom loader function
        - `_path` method with `(self, set_num)` input signature, returning
          path to file to load
        - `_get_set_nums` method
        - `_is_dataset` attribute

    **Modes of operation**:

        - *Directory*: one `batch`/`labels` per file. Filename includes `set_num`
          and `base_name`.
        - *Datasest*: all `batch`es / `labels` in one file, each batch accessed
          by `set_num`:

              - Mapping (.h5): keys must be string integers.
              - Numpy array: indexed directly, so shape must be
                `(n_batches, batch_size, *)`, i.e. `(batches, samples, *)`
    """
    _BUILTINS = {'numpy', 'numpy-memmap', 'numpy-lz4f',
                 'hdf5', 'csv'}  # TODO memmap

    def __init__(self, path, loader, dtype=None, batch_shape=None, base_name=None,
                 ext=None, filepaths=None):
        filepaths = self._validate_args(path, filepaths)
        self.path=path
        self.dtype=dtype
        self.batch_shape=batch_shape
        self.ext=ext
        self.base_name=base_name

        self._filepaths = filepaths
        self._filenames = [Path(p).name for p in filepaths]

        # safe to default to "dataset" even if file isn't formatted like one,
        # since one-batch file is a special case of many-batches file
        self._is_dataset = bool(len(filepaths) == 1)
        self._init_loader(loader, path, ext)

        if self._is_dataset and not os.path.isfile(self.path):
            self.path = self._filepaths[0]

    @property
    def load_fn(self):
        """Loads data given `set_num`. Is `data_loader` or `labels_loader` in
        :class:`DataGenerator`, set in :meth:`DataGenerator._infer_and_set_info`.

        **Setter**: if `loader` is

            - `None` passed to `__init__`, will set to string (one of builtins) in
              :meth:`_init_loader` based on `ext`, or to None if `path` is None.
            - String, will match to a supported builtin.
            - Function, will set to the function.
        """
        return self._load_fn

    @load_fn.setter
    def load_fn(self, loader):
        self._load_fn = self._get_loader(loader)

    def _get_loader(self, loader):
        def _validate_special_loaders(loader):
            if 'numpy-lz4f' in loader:
                if not IMPORTS['LZ4F']:
                    raise ImportError("`lz4framed` must be imported for "
                                      "`loader = 'numpy-lz4f'`")
                if self.batch_shape is None:
                    raise ValueError("'numpy-lz4f' loader requires "
                                     "`batch_shape` attribute set")

        supported = DataLoader._BUILTINS
        if isinstance(loader, LambdaType):  # custom
            setattr(self, loader.__name__, loader.__get__(self))
            loader = getattr(self, loader.__name__)
        elif loader is None:
            pass
        elif loader not in supported:
            raise ValueError(("unsupported loader '{}'; must be a custom "
                              "function, or one of {}").format(
                                  loader, ', '.join(supported)))
        else:
            _validate_special_loaders(loader)
            loader = getattr(self, loader.replace('-', '_') + '_loader')
        return loader

    def _init_loader(self, loader, path, ext):
        if path is None:
            loader = None
        elif loader is None:
            loader = {'.npy': 'numpy',
                      '.h5': 'hdf5',
                      '.csv': 'csv'}[ext]
        setattr(self, "load_fn", loader)  # setattr to avoid linter indexing

    #### LOADERS ##############################################################
    def _path(self, set_num):
        if self._is_dataset:
            return self.path
        return os.path.join(self.path, self.base_name + str(set_num) + self.ext)

    def numpy_loader(self, set_num):
        """For numpy arrays (.npy)."""
        if self._is_dataset:
            return np.load(self.path)[int(set_num)]
        return np.load(self._path(set_num))

    def hdf5_loader(self, set_num):
        """For hdf5 (.h5) files storing data one batch per file. `data_path`
        in :class:`DataGenerator` must contain more than one non-labels '.h5' file
        to default to this loader.
        """
        if self._is_dataset:
            with h5py.File(self.path, 'r') as hdf5_file:
                return hdf5_file[str(set_num)][:]
        with h5py.File(self._path(set_num), 'r') as hdf5_file:
            a_key = list(hdf5_file.keys())[0]  # only one should be present
            return hdf5_file[a_key][:]

    def csv_loader(self, set_num):
        """For .csv files (e.g. pandas.DataFrame)."""
        if self._is_dataset:
            return pd.read_csv(self.path)[set_num].to_numpy()
        return pd.read_csv(self._path(set_num)).to_numpy()

    def numpy_lz4f_loader(self, set_num):
        """For numpy arrays (.npy) compressed with `lz4framed`; see
        :func:`preprocessing.numpy_to_lz4f`.
        `self.data_dtype` must be original (save) dtype; if there's a
        mismatch, data of wrong value or shape will be decoded.

        Requires `data_batch_shape` / `labels_batch_shape` attribute to be set,
        as compressed representation omits shape info.
        """
        bytes_npy = lz4f.decompress(np.load(self._path(set_num)))
        data = np.frombuffer(bytes_npy, dtype=self.dtype
                             ).reshape(*self.batch_shape)
        if self._is_dataset:
            return data[set_num]
        return data

    #### Misc methods #########################################################
    def _get_set_nums(self):
        """Gets `set_nums_original` for :class:`DataGenerator`.

            - `_is_dataset`: will fetch from the single file based on `load_fn`.
            - Not `_is_dataset`: will fetch from filenames in `path`, delimiting
              with `base_name`. Ex: `data1.npy`, `data2.npy`, ..., and
              `base_name = 'data'` --> `set_nums = [1, 2, ...]`.
        """
        def _from_directory():
            def _sort_ascending(ls):
                return list(map(str, sorted(map(int, ls))))

            nums_from_dir = []
            for p in Path(self.path).iterdir():
                if p.suffix == self.ext and self.base_name in p.name:
                    clipped_name = p.stem.replace(self.base_name, '')
                    num = ''.join(c for c in clipped_name if c.isdigit())
                    nums_from_dir.append(num)
            return _sort_ascending(nums_from_dir)

        def _from_dataset():
            name = self.load_fn.__name__
            if name.startswith('numpy_lz4f'):
                bytes_npy = lz4f.decompress(np.load(self.path))
                data = np.frombuffer(bytes_npy, dtype=self.dtype,
                                     ).reshape(*self.batch_shape)
                return list(map(str, range(len(data))))
            elif name.startswith('numpy'):
                dataset = np.load(self.path)
                try:
                    len(dataset)
                except:
                    raise TypeError("unable to `len(dataset)`; the .npy file "
                                    "may be compressed: %s" % self.path)
                return list(map(str, range(len(dataset))))
            elif name.startswith('hdf5'):
                with h5py.File(self.path, 'r') as f:
                    return [num for num in list(f.keys())
                            if (num.isdigit() or
                                isinstance(num, (float, int)))]
            elif name.startswith('csv'):
                return list(pd.read_csv(self.path).keys())
            else:
                raise Exception("unsupported load_fn name: %s" %
                                self.load_fn.__name__ + "; must begin with "
                                "one of: numpy, numpy_lz4f, hdf5, csv. "
                                "Alternatively, provide `set_nums` to "
                                "`DataGenerator` at `__init__`")

        if self._is_dataset:
            return _from_dataset()
        return _from_directory()

    def _validate_args(self, path, filepaths):
        if (not isinstance(path, (str, type(None))) or
            (isinstance(path, str) and not
             (os.path.isfile(path) or os.path.isdir(path)))):
            raise TypeError("`path` must None, or string path to file / "
                            "directory; got %s" % path)
        if filepaths is None:
            if not os.path.isfile(path):
                raise Exception("if `filepaths` is not passed in, `path` must be "
                                "path to file to set `filepaths` to")
            filepaths = [path]
        elif not all(os.path.isfile(p) for p in filepaths):
            raise Exception("all entries in `filepaths` must be paths to "
                            "files; got:\n%s" % '\n'.join(filepaths))
        return filepaths
