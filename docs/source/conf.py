# Configuration file for the Sphinx documentation builder.
#
# For the full list of options see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Make the package importable without installing it, so that autosummary can
# introspect it both on Read the Docs and from a plain source checkout.
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "spectools"
copyright = "2020-2026, Vaishak Prasad"
author = "Vaishak Prasad"


def _get_version():
    """Read ``__version__`` out of ``spectools/__init__.py``.

    Parsed rather than imported so that a missing optional dependency can
    never break the version string.
    """

    init_file = os.path.join(
        os.path.abspath("../.."), "spectools", "__init__.py"
    )

    try:
        with open(init_file, "r", encoding="utf-8") as source:
            for line in source:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip("\"'")
    except OSError:
        pass

    return "unknown"


# The short X.Y version
version = _get_version()
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "recommonmark",
]

autosummary_generate = True
autosummary_imported_members = False

autodoc_docstring_signature = True
autodoc_default_options = {
    "members": True,
    # No "undoc-members": every public object now carries a docstring, so
    # the only things this would add are bare dataclass field descriptors,
    # which are already covered by their class's Attributes section --
    # and which numpydoc renders as malformed markup.
    "show-inheritance": True,
    "special-members": "__call__",
}

# The spherical and interpolation subpackages import heavy optional
# dependencies (MPI, waveformtools -> lalsuite, sxstools -> scri ->
# spinsfast). None of them are installable on a documentation builder, so
# mock them out: autodoc only needs the docstrings, not working imports.
autodoc_mock_imports = [
    "gmpy2",
    "h5py",
    "matplotlib",
    "mpi4py",
    "numba",
    "parallellib",
    "qlmtools",
    "scri",
    "spherical",
    "spinsfast",
    "sxstools",
    "sympy",
    "waveformtools",
]

# numpydoc insists on documenting every class member in its own stub page,
# which duplicates what autosummary already emits and asks for stub files
# that autosummary was never told to generate.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "style_nav_header_background": "blue",
    "navigation_depth": 4,
}

html_static_path = ["_static"]

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True
