How to ...?
***********

Change default configs
======================

Edit :mod:`deeptrain.util.configs`. 

    - Do **not** edit :mod:`deeptrain.util._default_configs`, this will break `DeepTrain`.
    - Arguments defined in :meth:`TrainGenerator.__init__` will override those specified in
      the configs (the defaults have no overlaps), so no point in specifying them in `configs`.

Run examples
============

`pip install deeptrain` excludes data by default. To acquire, you can:

    1. Build data by running scripts in `examples/preprocessing`. Or,
    2. Clone repository and copy-paste `examples/dir/data` into the pip-installed `deeptrain` directory.

With the data you can run the `.ipynb` with Jupyter or equivalent `.py` scripts in IPython. Note for docs notebook examples,
code isn't exact, and excludes some formatting irrelevant the examples (e.g. many used `os.environ['SCALEFIG'] = '.7'`).


Save train state
================

    1. Using :meth:`TrainGenerator.save`, which saves:
        
        - `TrainGenerator` attributes
        - DataGenerator` (both) attributes
        - Model state (layer weights, optimizer weights, and/or architecture)

    2. Using :meth:`TrainGenerator.checkpoint`, which saves what `.save()` saves, plus:
        
        - `TrainGenerator` report, made by :meth:`logging.generate_report`
        - Train & val history figure

    3. Saving behavior is configured for objects by respective attributes (defaults in\
    :mod:`~deeptrain.util._default_configs`):

        - `TrainGenerator`: `saveskip_list`
        - `DataGenerator` (for each): `saveskip_list`
        - `model`: `model_save_kw`, `model_save_weights_kw`, `optimizer_save_configs`
        - `Preprocessor` (of each `DataGenerator`): `saveskip_list`	   
	
Example in :doc:`examples/advanced`.


Load train state 
================

    1. Using :meth:`TrainGenerator.load`, which may load everything saved via :meth:`TrainGenerator.save`
    and :meth:`TrainGenerator.checkpoint`. 
	
    2. Loading behavior is configured for objects by respective attributes (defaults in\
    :mod:`~deeptrain.util._default_configs`):
    
        - `TrainGenerator`: `loadskip_list`
        - `DataGenerator` (for each): `loadskip_list`
        - `model`: `optimizer_load_configs`
        - `Preprocessor` (of each `DataGenerator`): `loadskip_list`

Example in :doc:`examples/advanced`.

		
Use custom train / evaluation function
======================================

Set `fit_fn` / `eval_fn`; see docs in :class:`TrainGenerator`.

