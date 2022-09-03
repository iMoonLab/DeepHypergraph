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
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))


def find_version(filename):
    """
    Find package version in file.
    """
    import re

    content = Path(filename).read_text()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


autodoc_mock_imports = ["torch", "numpy", "scipy", "optuna", "sklearn"]

# -- Project information -----------------------------------------------------

project = "DHG"
copyright = "2022, iMoonLab"
author = "iMoonLab"

# The full version, including alpha/beta/rc tags
release = find_version("../../dhg/__init__.py")

# custom configuration
# autodoc_member_order = 'bysource'

# -- General configuration ---------------------------------------------------

autodoc_typehints = "none"

# mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]
autosummary_generate = True

# for bibtex config
bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo_DHG_white.svg"
html_theme_options = {
    # "style_nav_header_background": "#9C27B0",
    "logo_only": True,
    "collapse_navigation": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

latex_use_xindy = False
latex_engine = "xelatex"
latex_elements = {
    "papersize": "a4paper",
    "utf8extra": "",
    "inputenc": "",
    "babel": r"""\usepackage[english]{babel}""",
    "preamble": r"""\usepackage{ctex}""",
}
latex_show_urls = "footnote"

def setup(app):
    app.add_css_file("css/dhg.css")

