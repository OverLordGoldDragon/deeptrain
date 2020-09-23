Framework Comparison
********************

DeepTrain vs Pytorch Lightning
==============================


.. raw:: html

    <style>
	    h3{margin-bottom: 0; padding-bottom: 0}
	    ul{margin: 0; padding: 0;}
        p{ margin: 0; padding: 0;}
        dl{margin: 0 !important; padding: 0 !important;}
        dd{margin: 0 10px !important; padding: 0 10px !important;}
        dt{margin: 0 !important; padding: 0 !important;}
		.ul{ margin-left: 20px;}
		.li{ margin-left: 20px; margin-top: 7px}
		.lin{margin-left: 30px; margin-top: 2px}
		.pl{font-size:18px; 
		    font-weight: bold;
			display: inline-block;
			transform: translateY(1px);
			}
    </style>
	
	<div class="ul">
	    <h3>Major</h3>
	    <div class="li"><span class="pl">+</span> <b>Resumable/interruptible</b>
		    <div class="lin">&#x2022; PL cannot be stopped and resumed mid-epoch without disrupting train/val loop & callbacks, DT can.</div>
			<div class="lin">&#x2022; <code>KeyboardInterrupt</code> any time, inspect model & train state as needeed, and resume.</div>
		</div>
	    <div class="li"><span class="pl">+</span> <b>Tracking state</b>
            <div class="lin">Much finer-grained tracking and control of internal train & data states</div>
        </div>
		<div class="li"><span class="pl">+</span> <b>Flexible batch_size</b>
		    <div class="lin">Set <code>batch_size</code> as integer/fraction multiple of that on file</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Faster SSD loading</b>
		    <div class="lin">Due to flexible <code>batch_size</code> (<a href="examples/misc/flexible_batch_size.html">example</a>)</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Reproducibility</b>
		    <div class="lin">DT's builtin callback sets seeds periodically, enabling reproducible training on epoch or batch level, 
			rather than only from very beginning of training (as with PL)
			</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Custom batch logic</b>
		    <div class="lin">&#x2022; Feed transformed batch to model arbitrary number of times before moving on to next</div>
		    <div class="lin">&#x2022; Control when "next" is</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Fully configurable saving & loading of model, optimizer, traingen, data generator</b>
		    <div class="lin">PL lacks attribute-specific configuring</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Windowed timeseries</b>
		    <div class="lin">&#x2022; variable number of windows per batch</div>
		    <div class="lin">&#x2022; window start increments</div>
		    <div class="lin">&#x2022; window overlaps</div>
		</div>

		<h3>Misc</h3>
		<div class="li"><span class="pl">+</span> <b>Model naming, image report generation</b> -- <a href="examples/misc/model_auto_naming.html">ex1</a>,
		                                                                                          <a href="examples/advanced.html#Inspect-generated-logs">ex2</a>
		</div>
		<div class="li"><span class="pl">+</span> <code>class_weights</code> <b>support</b>
		</div>
		<div class="li"><span class="pl">+</span> <b>Print metrics at batch-level</b>
		    <div class="lin">PL logs cumulatively, on epoch-level</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Shuffling options</b>
		    <div class="lin">Shuffle batches <i>and</i> samples within (across) batches 
			(<a href="deeptrain.html#deeptrain.data_generator.DataGenerator._make_group_batch_and_labels">docs</a>)</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Callbacks</b>
		    <div class="lin"><code>on_save</code> and <code>on_load</code> options for saving/loading callback object states</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Binary classifier threshold calibration </b>
		    <div class="lin">(DT finds optimal automatically)</div>
		</div>
		<div class="li"><span class="pl">+</span> <b>Best validation batch subset search (for e.g. ensembling)</b></div>
		<div class="li"><span class="pl">+</span> <b>Documentation</b>
		    <div class="lin">Methods and attributes are generally documented in greater scope & detail, with references to where each is used and what purpose they serve.</div>
		</div>
	</div>
	
	<hr>

	<div class="ul">
		<div class="li"><span class="pl">&ndash;</span> <b>TPU/Multi-GPU support</b>
		    <div class="lin">DT lacks builtin support (e.g. auto-conversion), but can still run if coded to</div>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>Gradient accumulation</b>
		    <div class="lin">DT lacks builtin support, but can implement by overriding <code>fit_fn</code></div>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>Learning rate finder</b>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>Learning rate scheduler</b>
		    <div class="lin">&#x2022; This was a design decision as updating LR externally is slower than by coding it into the optimizer's own loop</div>
		    <div class="lin">&#x2022; Can still update externally via callbacks</div>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>Examples scope</b>
		    <div class="lin">PL showcases more examples across various domains</div>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>ONNX export</b>
		    <div class="lin">DT lacks builtin support</div>
		</div>
		<div class="li"><span class="pl">&ndash;</span> <b>Support community</b>
		    <div class="lin">I am one, they are many. Bug reports, feature requests, etc. will be handled slower. Collaborators welcome.</div>
		</div>
    </div>
	
| 

