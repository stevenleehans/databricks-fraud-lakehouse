"""Pure-Python core of the PaySim fraud lakehouse PoC.

Importing this package must never require Spark, MLflow or a Databricks
connection -- the notebooks and the local pytest suite share it, and the tests
have to run on a laptop with no cluster attached.
"""

from . import (config, drift, features, impact, metrics, schema, serving,  # noqa: F401
               splits)

__all__ = ["config", "drift", "features", "impact", "metrics", "schema",
           "serving", "splits"]

__version__ = "0.1.0"
