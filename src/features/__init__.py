"""Feature registry: maps a full output feature name to a callable.

Every callable has signature `(group: src.streams.Group, config: dict) -> float`.
`group` is a single (user, session, q_id, r_t, version) cell; `config` is the
parsed config.yaml. Tier2/tier3 registries are split by r_t since e.g. code
stylometry doesn't apply to explanation text and vice versa -- the extraction
loop only calls the registry matching a group's own r_t.

To add a feature: write the function in the relevant tier module, add one
line here, and it is automatically picked up by src/extract.py and can be
toggled on/off by name via --features / --exclude-features.
"""
from __future__ import annotations

from src.features import tier1, tier2, tier3

TIER1_FEATURES: dict[str, callable] = dict(tier1.FEATURES)

TIER2_CODE_FEATURES: dict[str, callable] = dict(tier2.CODE_FEATURES)
TIER2_EXPLANATION_FEATURES: dict[str, callable] = dict(tier2.EXPLANATION_FEATURES)

TIER3_FEATURES: dict[str, callable] = dict(tier3.FEATURES)


def all_feature_names() -> list[str]:
    """Every fully-qualified feature name this registry can produce, for
    r_t in {code, explanation}, in the same "tier{n}_{r_t or 'llm'}_{name}"
    form used in the output CSV's `feature` column.
    """
    names = []
    for r_t in ("code", "explanation"):
        names += [f"tier1_{r_t}_{name}" for name in TIER1_FEATURES]
    names += [f"tier2_code_{name}" for name in TIER2_CODE_FEATURES]
    names += [f"tier2_explanation_{name}" for name in TIER2_EXPLANATION_FEATURES]
    names += [f"tier3_llm_{name}" for name in TIER3_FEATURES]
    return names
