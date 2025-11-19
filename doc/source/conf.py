# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ParkiPy"
copyright = "2025, Gabriel Kosmacher, Joar Bagge, Ziyu Du, George Biros"
author = "Gabriel Kosmacher, Joar Bagge, Ziyu Du, George Biros"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        "sphinx.ext.autodoc", 
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        #"sphinx.ext.linkcode",
]
autodoc_mock_imports = [
    "pykokkos", "cupy",
    "nvmath", "mpi4py",
]  # avoids crash for nonstandard module
napoleon_numpy_docstring=True
napoleon_custom_sections = [
    ("Post-Init Parameters", "params_style"),
]
maximum_signature_line_length=68

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Hook required by sphinx.ext.linkcode -----------------------------------
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://somesite/sourcerepo/%s.py" % filename
