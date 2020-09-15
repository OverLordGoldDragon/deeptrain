# Configuration file for the Sphinx documentation builder.

#### Path setup ##############################################################

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import sys
from pathlib import Path

confdir = Path(__file__).parent
sys.path.insert(0, str(Path(str(confdir.parents[2]), "see-rnn")))  # for local
sys.path.insert(0, str(confdir))             # conf.py dir
sys.path.insert(0, str(confdir.parents[0]))  # docs dir
sys.path.insert(0, str(confdir.parents[1]))  # package rootdir

#### Project info  ###########################################################
import deeptrain

project = 'DeepTrain'
author = deeptrain.__author__
copyright = deeptrain.__copyright__

# The short X.Y version
version = deeptrain.__version__
# The full version, including alpha/beta/rc tags
release = deeptrain.__version__

#### General configs #########################################################

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


##### HTML output configs ####################################################

html_sidebars = { '**': [
    'globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# CSS to customize HTML output
html_css_files = [
    'style.css',
]

html_favicon = '_images/favicon.png'

# Make "footnote [1]_" appear as "footnote[1]_"
trim_footnote_reference_space = True

# ReadTheDocs sets master doc to index.rst, whereas Sphinx expects it to be
# contents.rst:
master_doc = 'index'

# make `code` code, instead of ``code``
default_role = 'literal'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

##### Theme configs ##########################################################
import sphinx_rtd_theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'

##### Autodoc configs ########################################################
from importlib import import_module
from docutils.parsers.rst import Directive
from docutils import nodes
from sphinx import addnodes
from inspect import getsource


# document lists, tuples, dicts, np.ndarray's exactly as in source code
class PrettyPrintIterable(Directive):
    required_arguments = 1

    def run(self):
        def _get_iter_source(src, varname):
            # 1. identifies target iterable by variable name, (cannot be spaced)
            # 2. determines iter source code start & end by tracking brackets
            # 3. returns source code between found start & end
            start = end = None
            open_brackets = closed_brackets = 0
            for i, line in enumerate(src):
                if line.startswith(varname):
                    if start is None:
                        start = i
                if start is not None:
                    open_brackets   += sum(line.count(b) for b in "([{")
                    closed_brackets += sum(line.count(b) for b in ")]}")

                if open_brackets > 0 and (open_brackets - closed_brackets == 0):
                    end = i + 1
                    break
            return '\n'.join(src[start:end])

        module_path, member_name = self.arguments[0].rsplit('.', 1)
        src = getsource(import_module(module_path)).split('\n')
        code = _get_iter_source(src, member_name)

        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'

        return [addnodes.desc_name(text=member_name),
                addnodes.desc_content('', literal)]


# Document module / class methods in order of writing (rather than alphabetically)
autodoc_member_order = 'bysource'

def skip(app, what, name, obj, would_skip, options):
    # do not pull sklearn metrics docs in deeptrain.metrics
    if getattr(obj, '__module__', '').startswith('sklearn.metrics'):
        return True
    # include private methods (but not magic) if they have documentation
    if name.startswith('_') and getattr(obj, '__doc__', '') and (
            '__%s__' % name.strip('__') != name):
        return False
    return would_skip

#### nbsphinx configs###############################################
nbsphinx_thumbnails = {
    'examples/misc/timeseries': '_images/ecg.png',
}

#### Reference by alias configs ###############################################
from sphinx.addnodes import pending_xref
from docutils.nodes import Text

# alias ref is mapped to a pair (real ref, text to render)
# {alias: (ref, render)}
# type reversed, nicer alignment: {refp: alias}, where `refp` will prepend self
# to `alias `, unless `refp[-1] == '#'`
#
# Ex:
# ('data_generator', 'DataGenerator') ->
# {'DataGenerator': ('deeptrain.data_generator.DataGenerator', 'DataGenerator()')}
#
# ('data_generator*', 'DataGenerator') ->
# {'DataGenerator': ('deeptrain.data_generator.DataGenerator', 'DataGenerator')}
#
# ('*', 'util.data_loaders') ->
# {'util.data_loaders': ('deeptrain.util.data_loaders', 'util.data_loaders')}
reftarget_aliases = [
    ('train_generator', 'TrainGenerator._train_postiter_processing'),
    ('train_generator', 'TrainGenerator._val_postiter_processing'),
    ('train_generator', 'TrainGenerator._init_and_validate_kwargs'),
    ('train_generator', 'TrainGenerator._print_iter_progress'),
    ('train_generator', 'TrainGenerator._print_train_progress'),
    ('train_generator', 'TrainGenerator._print_val_progress'),
    ('train_generator', 'TrainGenerator._print_progress'),
    ('train_generator', 'TrainGenerator._metric_name_to_alias'),
    ('train_generator', 'TrainGenerator._alias_to_metric_name'),
    ('train_generator', 'TrainGenerator.reset_validation'),
    ('train_generator', 'TrainGenerator.check_health'),
    ('train_generator', 'TrainGenerator._on_val_end'),
    ('train_generator', 'TrainGenerator.__init__'),
    ('train_generator', 'TrainGenerator.get_data'),
    ('train_generator', 'TrainGenerator.train'),
    ('train_generator', 'TrainGenerator.validate'),
    ('train_generator', 'TrainGenerator'),
    ('data_generator',  'DataGenerator'),
    ('data_generator',  'DataGenerator.reset_state'),
    ('data_generator',  'DataGenerator.get'),
    ('data_generator',  'DataGenerator.advance_batch'),
    ('data_generator',  'DataGenerator.on_epoch_end'),
    ('data_generator',  'DataGenerator._set_preprocessor'),
    ('data_generator',  'DataGenerator._make_group_batch_and_labels'),
    ('data_generator',  'DataGenerator._init_and_validate_kwargs'),
    ('data_generator',  'DataGenerator._infer_and_set_info'),
    ('data_generator',  'DataGenerator._get_next_batch'),
    ('data_generator',  'DataGenerator._set_data_loader'),
    ('util.saving',     'checkpoint'),
    ('util',            'misc._make_plot_configs_from_metrics'),
    ('',                'preprocessing.numpy_to_lz4f'),
    ('*',    'util.data_loaders'),
    ('util', 'data_loader.numpy_loader'),
    ('util', 'logging.generate_report'),
    ('util.preprocessors',  'Preprocessor'),
    ('util.preprocessors',  'GenericPreprocessor'),
    ('util.preprocessors',  'TimeseriesPreprocessor'),
    ('util.preprocessors',  'Preprocessor.process'),
    ('util.data_loaders',   'DataLoader'),
    ('util.data_loaders',   'DataLoader.load_fn'),
    ('util.data_loaders',   'DataLoader._get_set_nums'),
    ('util.data_loaders',   'DataLoader.numpy_loader'),
    ('util.experimental',   'deepcopy_v2'),
    ('util._default_configs*', '_DEFAULT_PLOT_CFG'),
    ('util.configs*',          '_PLOT_CFG'),
]
# make into dict, reverse
reftarget_aliases = {v: k for k, v in reftarget_aliases}
ra = {}
for k, v in reftarget_aliases.items():
    pre_v  = bool('#' not in v[-2:])  # flag to not prepend `v` to `k`
    no_par = bool('*' in v[-2:])      # flag to not append `()` to `k`
    if no_par and not pre_v:
        v = v[:-2]  # drop '#' and '*'
    elif no_par or not pre_v:
        v = v[:-1]  # drop '#' or  '*'

    if pre_v:
        if v == '':  # i.e. v == '*' originally; avoid e.g. "deeptrain..util"
            v = k
        else:
            v += '.' + k

    if no_par:
        ra[k] = ('deeptrain.' + v, k)
    else:
        ra[k] = ('deeptrain.' + v, k + '()')

# special cases not fitting above pattern
ra.update({
    'TrainGenerator.save': ('deeptrain.util.saving.save',
                            'TrainGenerator.save()'),
    'TrainGenerator.load': ('deeptrain.util.saving.load',
                            'TrainGenerator.load()'),
    'TrainGenerator.checkpoint': ('deeptrain.util.saving.checkpoint',
                                  'TrainGenerator.checkpoint()'),
})
reftarget_aliases = ra


def resolve_internal_aliases(app, doctree):
    pending_xrefs = doctree.traverse(condition=pending_xref)
    for node in pending_xrefs:
        alias = node.get('reftarget', None)
        if alias is not None and alias in reftarget_aliases:
            real_ref, text_to_render = reftarget_aliases[alias]
            # this will resolve the ref
            node['reftarget'] = real_ref

            # this will rewrite the rendered text:
            # find the text node child
            text_node = next(iter(node.traverse(lambda n: n.tagname == '#text')))
            # remove the old text node, add new text node with custom text
            text_node.parent.replace(text_node, Text(text_to_render, ''))

###############################################################################

def setup(app):
    app.add_stylesheet("style.css")
    app.add_directive('pprint', PrettyPrintIterable)
    app.connect('doctree-read', resolve_internal_aliases)
    app.connect("autodoc-skip-member", skip)
