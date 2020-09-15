from abc import ABCMeta, abstractmethod
from ._backend import WARN


class Preprocessor(metaclass=ABCMeta):
    """Abstract base class for preprocessors, outlining required methods for
    operability with :class:`DataGenerator`.

    The following attributes are "synched" with :class:`DataGenerator`:
    `batch_loaded`, `batch_exhausted`, `slices_per_batch`, `slice_idx`. Setter
    and getter are implemented to set and get these attributes from the
    preprocessor, so they are always same for `Preprocessor` and `DataGenerator`.
    """
    def __new__(cls, *args, **kwargs):
        """Ensures preprocessors have expected attributes defined.
        """
        cls.batch_loaded = False
        cls.batch_exhausted = False
        cls.slices_per_batch = None
        cls.slice_idx = None
        cls.loadskip_list = []
        return object.__new__(cls)

    @abstractmethod
    def process(self, batch, labels):
        """Required to implement; must return `(batch, labels)`. Can
        apply arbitrary preprocessing steps, or return as-is.
        Is called within :meth:`DataGenerator.get`.
        """
        pass

    def update_state(self):
        """Optional to implement; must involve setting `batch_exhausted` and
        `batch_loaded` attributes to True or False.
        """
        self.reset_state()

    def reset_state(self):
        """Optional to implement. Can be used to reset attributes specific
        to the preprocessor.
        Is called within :meth:`DataGenerator.reset_state`.
        """
        self.batch_exhausted = True
        self.batch_loaded = False

    def on_epoch_end(self, epoch):
        """Optional to implement. Can be used to do things at end of epoch.
        Is called within :meth:`DataGenerator.on_epoch_end`, which is
        called by `_on_epoch_end` within
        :meth:`TrainGenerator._train_postiter_processing` or
        :meth:`TrainGenerator._val_postiter_processing`.
        """
        pass

    def _validate_configs(self):
        """Internal method to validate `slices_per_batch` in
        :meth:`DataGenerator._set_preprocessor`.
        """
        spb = self.slices_per_batch
        if not (spb is None or (isinstance(spb, int) and spb >= 1)):
            raise ValueError("`slices_per_batch` must be None or int >= 1, "
                             "got: %s" % spb)


class TimeseriesPreprocessor(Preprocessor):
    """Stateful preprocessor breaking up batches into "windows".

    Arguments:
        window_size: int
            Length of each window (dim1), or number of timesteps per slice.
        slide_size: int
            Number of timesteps by which to slide the window.
        start_increments: int
            Number of timesteps by which to increment each window when fetching.
        loadskip_list: dict / None
            Attributes to skip on :meth:`TrainGenerator.load`. Defaults to
            `['start_increments', 'window_size', 'slide_size']`.

    A "slice" here is a "window", and `slices_per_batch` is the number of such
    windows per batch.

    **Examples**:

    Each window in `windows` is from calling :meth:`_next_window`; changing
    `start` & `end` requires calling :meth:`update_state`.

    >>> batch.shape == (32, 100, 4)
    ...
    >>> window_size, slide_size, start_increment = (25, 25, 0)
    >>> slices_per_batch == 4
    >>> windows == [batch[:, :25],     # slice_idx = 0 (window 1)
    ...             batch[:, 25:50],   # slice_idx = 1 (window 2)
    ...             batch[:, 50:75],   # slice_idx = 2 (window 3)
    ...             batch[:, 75:100]]  # slice_idx = 3 (window 4)
    ...
    >>> window_size, slide_size, start_increment = (25, 25, 10)
    >>> slices_per_batch == 3
    >>> windows == [batch[:, 10:35],   # slice_idx = 0 (window 1)
    ...             batch[:, 35:60],   # slice_idx = 1 (window 2)
    ...             batch[:, 60:85]]   # slice_idx = 2 (window 3)
    ...
    >>> window_size, slide_size, start_increment = (25, 10, 0)
    >>> slices_per_batch == 8
    >>> windows == [batch[:, :25],     # slice_idx = 0 (window 1)
    ...             batch[:, 10:35],   # slice_idx = 1 (...)
    ...             batch[:, 20:45],   # slice_idx = 2
    ...             batch[:, 30:55],   # slice_idx = 3
    ...             batch[:, 40:65],   # slice_idx = 4
    ...             batch[:, 50:75],   # slice_idx = 5
    ...             batch[:, 60:85],   # slice_idx = 6
    ...             batch[:, 70:95]]   # slice_idx = 7 (window 8)
    """
    def __init__(self, window_size,
                 slide_size=None,
                 start_increments=None,
                 loadskip_list=None):
        self.window_size=window_size
        self.slide_size=slide_size or window_size
        self.start_increments=start_increments

        # can't know before `process`; set to truthy to identify self as
        # sliced preprocessor (e.g. in misc._validate_traingen_configs)
        self.slices_per_batch = 1

        self._start_increment = 0
        self._maybe_set_start_increment(epoch=0)
        self.reset_state()

        self.loadskip_list=loadskip_list or [
            'start_increments', 'window_size', 'slide_size']

    def process(self, batch, labels):
        """Return next `batch` window, and unchanged `labels`."""
        if self.slice_idx == 0:
            # ensure number of windows accurate for every new batch
            self._batch_timesteps = batch.shape[1]
            self._set_slices_per_batch()
        return self._next_window(batch), labels

    def _next_window(self, batch):
        """Fetches temporal slice according to `window_size`, `slide_size`,
        `start_increment`, and `slice_idx`;
        See :class:`TimeseriesPreprocessor` for examples."""
        start = self.slice_idx * self.slide_size + self.start_increment
        end   = start + self.window_size
        return batch[:, start:end]

    def reset_state(self):
        """Set `slice_idx = 0.`"""
        self.slice_idx = 0

    def on_epoch_end(self, epoch):
        """Update `slices_per_batch`, and `start_increment` if `start_increments`
        is not None."""
        self._maybe_set_start_increment(epoch)

    def update_state(self):
        """Increment `slice_idx` by 1; if `slice_idx == slices_per_batch`,
        set `batch_exhausted = True`, `batch_loaded = False`.
        """
        self.slice_idx += 1
        if self.slice_idx == self.slices_per_batch:
            self.batch_exhausted = True
            self.batch_loaded = False

    #### Helper methods #######################################################
    def _set_slices_per_batch(self):
        self.slices_per_batch = 1 + (
            self._batch_timesteps - self.window_size - self.start_increment
            ) // self.slide_size

    def _maybe_set_start_increment(self, epoch):
        if self.start_increments is not None:
            self.start_increment = self.start_increments[
                epoch % len(self.start_increments)]

    @property
    def start_increment(self):
        """Sliding window start increment; see `help(TimeseriesPreprocessor)`."""
        return self._start_increment

    @start_increment.setter
    def start_increment(self, value):
        def _validate(value):
            if not isinstance(value, int):
                raise ValueError("`start_increment` must be set to "
                                 "integer (got: %s)" % value)
            if value not in self.start_increments:
                print(WARN,
                      ("setting `start_increment` to {}, which is not in "
                       "`start_increments` ({})").format(
                           value, ", ".join(map(str, self.start_increments))))

        if self.start_increments is not None:
            _validate(value)
            self._start_increment = value
        else:
            raise ValueError("setting `start_increment` is unsupported if "
                             "`start_increments` is None")


class GenericPreprocessor(Preprocessor):
    """Minimal Preprocessor; does nothing to `batch` or `labels`, but maintains
    `batch_exhausted` and `batch_loaded` logic.
    """
    def __init__(self, loadskip_list=None):
        self.loadskip_list=loadskip_list or []
        self.reset_state()

    def process(self, batch, labels):
        """Return `batch` and `labels` as-is."""
        return batch, labels
