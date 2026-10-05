# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dcatoolkit'
copyright = '2026, Raheel Syed Ahmed'
author = 'Raheel Syed Ahmed'
# Read from the installed package, so the docs always match pyproject.toml's version.
release = _version('dcatoolkit')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Pull docstrings from the code.
    'sphinx.ext.autosummary',  # Generate one page per class.
    'numpydoc',                # Render numpydoc-style docstrings.
    'sphinx.ext.intersphinx',  # Link types like numpy.ndarray to their own documentation.
    'sphinx.ext.viewcode',     # Add "[source]" links.
    'myst_parser',             # Write pages in Markdown, and include CHANGELOG.md directly.
]

# Give Markdown headings (down to ###) link targets, so links like CHANGELOG.md's #upgrading-from-02x work.
myst_heading_anchors = 3

autosummary_generate = True
# Document every public method on each class page, including ones inherited from StructureInformation.
autodoc_default_options = {'members': True, 'inherited-members': True}
# autodoc already lists the members, so skip numpydoc's duplicate member table.
numpydoc_show_class_members = False
# Turn parameter and return types (e.g. numpy.ndarray) into links; leave the plain words in type descriptions alone.
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {'of', 'or', 'optional', 'default', 'excluding', 'shape'}
numpydoc_xref_aliases = {'Iterable': 'collections.abc.Iterable'}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'biotite': ('https://www.biotite-python.org/latest', None),
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
