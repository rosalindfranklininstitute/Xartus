# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
from pathlib import Path
from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files
from xartus.api import data_convert

sys.path.insert(0, str(Path("..").absolute()))

project = "Xartus"
copyright = "2026, Duncan McDougall (duncan.mcdougall@rfi.ac.uk)"  # noqa: A001
author = "Duncan McDougall (duncan.mcdougall@rfi.ac.uk)"
release = "v0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_default_options = {
    "special-members": "__enter__, __exit__, __getitem__, __setitem__, __contains__",
    "undoc-members": False,
    "show-inheritance": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# --- Generate test data ---

man_file = data_files()["man1"]
man_data = ManData()
man_data_source = Man2DDataSource(
    man_data,
    multipliers=dict(x=0.1, y=0.1, mz=0.1, time=1.0, error=1.0),
)
filename = Path(__file__).resolve().parent / "test_data.nxs"
if not filename.exists():
    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=filename,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})


os.environ["DOCS_SOURCE_DIR"] = str(filename.parent)
