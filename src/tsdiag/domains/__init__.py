from .bearing import BEARING_PACK
from .bearing_runner import BearingDiagnosticPipeline
from .process import PROCESS_PACK
from .process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult
from .wind_scada import WIND_SCADA_PACK
from .battery import BATTERY_PACK
from .turbofan import TURBOFAN_PACK
from .transformer import TRANSFORMER_PACK

DOMAIN_PACKS = {
    pack.key: pack
    for pack in (
        BEARING_PACK,
        PROCESS_PACK,
        WIND_SCADA_PACK,
        BATTERY_PACK,
        TURBOFAN_PACK,
        TRANSFORMER_PACK,
    )
}


def get_domain_pack(key: str):
    try:
        return DOMAIN_PACKS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown domain pack {key!r}. Available: {sorted(DOMAIN_PACKS)}") from exc


# Import adapters after DOMAIN_PACKS/get_domain_pack are defined to avoid circular
# imports through the shared core runtime.
from .runtime_adapters import run_bearing_request, run_process_request


__all__ = [
    "BEARING_PACK",
    "BearingDiagnosticPipeline",
    "PROCESS_PACK",
    "ProcessDiagnosticPipeline",
    "ProcessDiagnosticResult",
    "WIND_SCADA_PACK",
    "BATTERY_PACK",
    "TURBOFAN_PACK",
    "TRANSFORMER_PACK",
    "DOMAIN_PACKS",
    "get_domain_pack",
    "run_bearing_request",
    "run_process_request",
]
