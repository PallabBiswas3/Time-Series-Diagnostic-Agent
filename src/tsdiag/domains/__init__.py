from .bearing import BEARING_PACK
from .process import PROCESS_PACK
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


__all__ = [
    "BEARING_PACK",
    "PROCESS_PACK",
    "WIND_SCADA_PACK",
    "BATTERY_PACK",
    "TURBOFAN_PACK",
    "TRANSFORMER_PACK",
    "DOMAIN_PACKS",
    "get_domain_pack",
]
