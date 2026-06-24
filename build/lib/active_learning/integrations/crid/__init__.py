"""CRID integration exports."""

__all__ = [
    "CridExportResult",
    "CridImageProviderSource",
    "CridPool",
    "CridSource",
    "export_selection",
    "get_export_urls",
]


def __getattr__(name):
    if name == "CridImageProviderSource":
        from active_learning.integrations.crid.provider_source import (
            CridImageProviderSource,
        )

        return CridImageProviderSource
    if name in {"CridPool", "CridSource"}:
        from active_learning.integrations.crid.source import CridPool, CridSource

        return {"CridPool": CridPool, "CridSource": CridSource}[name]
    if name in {"CridExportResult", "export_selection", "get_export_urls"}:
        from active_learning.integrations.crid.export import (
            CridExportResult,
            export_selection,
            get_export_urls,
        )

        return {
            "CridExportResult": CridExportResult,
            "export_selection": export_selection,
            "get_export_urls": get_export_urls,
        }[name]
    raise AttributeError(name)
