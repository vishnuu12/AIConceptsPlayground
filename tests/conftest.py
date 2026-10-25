"""
Shared test setup.

The demos print a readable trace and (optionally) save a PNG. Tests do not want
either, so plotting is switched off before any demo module is imported. The demos
read this variable inside `data.save`, which means it works no matter when the
import happens.
"""

import os

os.environ.setdefault("AICONCEPTS_NO_PLOTS", "1")
