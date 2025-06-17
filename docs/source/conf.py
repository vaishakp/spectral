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

import sphinx
from recommonmark.parser import CommonMarkParser

# import spectools
# from spectral import get_version

# -- Project information -----------------------------------------------------

# Load the package into pythonpath
cwd = os.getcwd()
print("CurrWD", cwd)

pack_path = cwd + "/../../"
sys.path.append(pack_path)

print("Pythonpath:", sys.path)
# with open('../../public/date.txt', 'r') as f:
#    proj_vers = f.readline()

# Fetch the latest commit version
dvers = os.popen("git log -1 --date=short | grep Date").read()

print("Date fetched", dvers)
print("Version string", dvers[8:-1])
proj_vers = dvers
print("Parsed version:", proj_vers)
spectral_version = proj_vers  # get_version()

# -- Project information -----------------------------------------------------

project = "spectral"
copyright = "2020, Vaishak Prasad"
author = "Vaishak Prasad"

# The short X.Y version
version = spectral_version
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    #    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    #    "sphinx.ext.imgmath",
    #    "myst_parser"
    "recommonmark",
]


autosummary_generate = True

autodoc_docstring_signature = True
if sphinx.version_info < (1, 8):
    autodoc_default_flags = ["members", "undoc-members"]
else:
    autodoc_default_options = {
        "members": None,
        "undoc-members": None,
        "special-members": "__call__",
    }

# -- Try to auto-generate numba-decorated signatures -----------------

# import numba
import inspect


# def process_numba_docstring(app, what, name, obj, options, signature, return_annotation):
#    if type(obj) is not numba.core.registry.CPUDispatcher:
#        return (signature, return_annotation)
#    else:
#        original = obj.py_func
#        orig_sig = inspect.signature(original)

#        if (orig_sig.return_annotation) is inspect._empty:
#            ret_ann = None
#        else:
#            ret_ann = orig_sig.return_annotation.__name__
#
#        return (str(orig_sig), ret_ann)


# def setup(app):
#    app.connect("autodoc-process-signature", process_numba_docstring)


# --------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
if sphinx.version_info < (1, 8):
    source_parsers = {
        ".md": CommonMarkParser,
    }
    source_suffix = [".rst", ".md"]
else:
    source_suffix = {
        ".rst": "restructuredtext",
        ".txt": "markdown",
        ".md": "markdown",
    }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_style = 'css/my_theme.css'

html_theme_options = {
    "analytics_id": "G-XXXXXXXXXX",  #  Provided by Google in your dashboard
    #'analytics_anonymize_ip': False,
    #'logo_only': False,
    "display_version": True,
    #'prev_next_buttons_location': 'bottom',
    #'style_external_links': False,
    #'vcs_pageview_mode': '',
    "style_nav_header_background": "blue",
    # 	"body_max_width" : "70%"
    # Toc options
    #'collapse_navigation': True,
    #'sticky_navigation': True,
    #'navigation_depth': 4,
    #'includehidden': True,
    #'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
