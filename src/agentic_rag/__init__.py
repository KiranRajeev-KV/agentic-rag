import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

__all__ = ["__version__"]

__version__ = "0.1.0"

warnings.filterwarnings(
    "ignore",
    message=(
        r"The default value of `allowed_objects` will change in a future version\..*"
    ),
    category=LangChainPendingDeprecationWarning,
    module=r"langgraph\.cache\.base(\.__init__)?",
)
