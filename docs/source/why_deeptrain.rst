Why DeepTrain?
**************

DeepTrain is founded on **control** and **introspection**: full knowledge and manipulation of the train state.

What does it do?
================

Abstract away boilerplate train loop and data loading code, *without* making it into a black box. Code is written intuitively and fully documented.
Everything about the train state can be seen via *dedicated attributes*; which batch is being fit and when, how long until an epoch ends, intermediate metrics, etc.

DeepTrain is *not* a "wrapper" around TF; while currently only supporting TF, fitting and data logic is framework-agnostic.

When is it suitable (and not)?
==============================

Training *few* models *thoroughly*: closely tracking model and train attributes to debug performance and inform next steps.

DeepTrain is *not* for models that take under an hour to train, or for training hundreds of models at once.

Features
========

Train Loop
----------

  - **Control**: iteration-, batch-, epoch-level customs
  - **Resumability**: interrupt-protection, can pause mid-training -- `ex <examples/introspection/internals.html#Interrupts>`_
  - **Tracking & reproducibility**: save & load model, train state, random seeds, and hyperparameter info
  - **Callbacks** at any stage of training or validation -- :doc:`ex <examples/callbacks/basic>`

Data Pipeline
-------------

  - **AutoData**: need only path to directory, the rest is inferred (but can customize)
  - **Faster SSD loading**: load larger batches to maximize read speed utility
  - **Flexible batch size**: can differ from that of loaded files, will split/combine  -- :doc:`ex <examples/misc/flexible_batch_size>`
  - **Stateful timeseries**: splits up a batch into windows, and `reset_states()` (RNNs) at end -- :doc:`ex <examples/misc/timeseries>`
  - **Iter-level preprocessor**: pass batch & labels through :class:`Preprocessor` before feeding to model -- :doc:`ex <examples/misc/preprocessor>`
  - **Loader function**: define custom data loader for any file extension, handled by :class:`DataLoader`
  
Introspection
-------------

  - **Data**: batches and labels are enumerated by "set nums"; know what's being fit and when
  - **Model**: auto descriptive naming; gradients, weights, activations visuals -- :doc:`ex1 <examples/misc/model_auto_naming>`, :doc:`ex2 <examples/callbacks/mnist>`
  - **Train state**: single-image log of key attributes & hyperparameters for easy reference -- `ex <examples/advanced.html#Inspect-generated-logs>`_

Utilities
---------

  - **Preprocessing**: batch-making and format conversion methods -- `docs <deeptrain.html#module-deeptrain.preprocessing>`_
  - **Calibration**: classifier prediction threshold; best batch subset selection (for e.g. ensembling) -- `docs <deeptrain.util.html#module-deeptrain.util.searching>`_
  - **Algorithms**: convenience methods for object inspection & manipulation -- `docs <deeptrain.util.html#module-deeptrain.util.algorithms>`_
  - **Callbacks**: reusable methods with other libraries supporting callbacks -- `docs <deeptrain.html#module-deeptrain.callbacks>`_

List not exhaustive; for application-specific features, see :doc:`Framework Comparison <framework_comparison>`.
