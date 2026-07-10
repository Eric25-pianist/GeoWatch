"""GeoWatch package metadata and public constants."""

from __future__ import annotations

import os

# Keep Windows geospatial/scikit-learn runs stable on laptops where mixed
# OpenMP runtimes can otherwise over-subscribe CPU threads or terminate silently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

__version__ = "0.2.0"
