# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'PVAnalytics'
copyright = '2020-2022, pvlib'
author = 'pvlib'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx_gallery.gen_gallery',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'examples']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/pvlib/pvanalytics",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pvanalytics/",
            "icon": "fab fa-python",
        },
    ],
    # "use_edit_page_button": True,
    "show_toc_level": 1,
    "footer_items": ["copyright", "sphinx-version", "sidebar-ethical-ads"],
    "left_sidebar_end": [],
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

extlinks = {
    'issue': (
        'https://github.com/pvlib/pvanalytics/issues/%s',
        'GH'),
    'pull': (
        'https://github.com/pvlib/pvanalytics/pull/%s',
        'GH'),
    'wiki': (
        'https://github.com/pvlib/pvanalytics/wiki/%s',
        'wiki '),
    'doi': ('http://dx.doi.org/%s', 'DOI: '),
    'ghuser': ('https://github.com/%s', '@')
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'pvlib': ('https://pvlib-python.readthedocs.io/en/stable/', None),
}


# settings for sphinx-gallery
sphinx_gallery_conf = {
    'examples_dirs': ['examples'],  # location of gallery scripts
    'gallery_dirs': ['generated/gallery'],  # location of generated output
    # sphinx-gallery only shows plots from plot_*.py files by default:
    'filename_pattern': re.escape(os.sep),

    # directory where function/class granular galleries are stored
    'backreferences_dir': 'generated/gallery_backreferences',

    # Modules for which function/class level galleries are created. In
    # this case only pvanalytics currently.  must be tuple of str
    'doc_module': ('pvanalytics',),
}
