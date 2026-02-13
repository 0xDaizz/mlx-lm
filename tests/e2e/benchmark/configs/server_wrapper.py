#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Thin wrapper to launch mlx_lm_server via mlx.launch.

mlx.launch expects a Python script file as its positional argument, not
``python -m module``.  This wrapper bridges the gap: mlx.launch invokes
this file, which in turn calls the server's ``main()`` entry-point.
Command-line arguments after ``--`` are forwarded via sys.argv as usual.
"""

import sys

sys.path.insert(0, "/Users/hw/mlx-lm-server")

from mlx_lm_server.__main__ import main

main()
