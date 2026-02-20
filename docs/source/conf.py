from __future__ import annotations

import sys
from pathlib import Path

# Repo layout: <repo>/src/fast_lisa_subtraction/...
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

project = "Fast Lisa Subtraction"
author = "Federico De Santi"
copyright = "2026, Federico De Santi"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    #"myst_parser",
    "myst_nb",
    "sphinx_copybutton",
]

nb_execution_mode = "off"


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# exclude_patterns += [
#     "api/fast_lisa_subtraction.rst",
#     "api/priors.rst",
#     "api/simulation.rst",
#     "api/utils.rst",
# ]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
