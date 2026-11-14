"""BRTP geometry defaults. Frozen plans, not this shim, own training recipes."""

import math
from numbers import Real
from string import Formatter

from yacs.config import CfgNode


def model_options() -> CfgNode:
    """Full geometry and opt-in controls used by the retained ablations."""
    return CfgNode(
        {
            "TEMPLATES": [
                "a photo of a {}.",
                "a satellite image of a {}.",
                "an aerial image of a {}.",
            ],
            "BANDWIDTH": 10.0,
            "RIDGE": 0.1,
            "MAX_RESIDUAL_NORM": 0.25,
            "TRANSPORT": True,
            "COVERAGE": True,
            "RESIDUAL_BOUND": True,
            "COVERAGE_MODE": "adaptive",
            "AUXILIARY_SELF_DONOR": False,
            "TANGENT_PROJECTION": True,
            "AUXILIARY_WRONG_DONOR": False,
        }
    )


def validate_template(template: str) -> None:
    """Require one positional class slot and no other formatting fields."""
    if not isinstance(template, str):
        raise ValueError("Exactly one class slot is required per template")
    try:
        fields = [
            (field, spec, conversion)
            for _, field, spec, conversion in Formatter().parse(template)
            if field is not None
        ]
    except ValueError as error:
        raise ValueError("Invalid class template") from error
    if fields != [("", "", None)]:
        raise ValueError("Exactly one plain {} class slot is required per template")


def validated_model_options(options: CfgNode) -> CfgNode:
    """Clone and validate geometry, including older frozen option schemas."""
    options = options.clone()
    defaults = model_options()
    for key in (
        "RESIDUAL_BOUND",
        "COVERAGE_MODE",
        "AUXILIARY_SELF_DONOR",
        "TANGENT_PROJECTION",
        "AUXILIARY_WRONG_DONOR",
    ):
        if key not in options:
            setattr(options, key, defaults[key])
    for key in (
        "TRANSPORT",
        "COVERAGE",
        "RESIDUAL_BOUND",
        "AUXILIARY_SELF_DONOR",
        "TANGENT_PROJECTION",
        "AUXILIARY_WRONG_DONOR",
    ):
        if not isinstance(options[key], bool):
            raise TypeError(key + " must be a boolean")
    for key in ("BANDWIDTH", "RIDGE", "MAX_RESIDUAL_NORM"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(key + " must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be positive and finite")
    if options.COVERAGE_MODE not in ("adaptive", "fixed_half"):
        raise ValueError("Unknown coverage mode")
    if options.AUXILIARY_SELF_DONOR and options.AUXILIARY_WRONG_DONOR:
        raise ValueError("Self-donor and wrong-exclusion controls are mutually exclusive")
    if not isinstance(options.TEMPLATES, (list, tuple)) or not options.TEMPLATES:
        raise ValueError("A nonempty sequence of class templates is required")
    for template in options.TEMPLATES:
        validate_template(template)
    return options


def get_cfg() -> CfgNode:
    """Historical Full runner radius assertion; this is not a train recipe."""
    cfg = CfgNode()
    cfg.TRAINER = CfgNode()
    cfg.TRAINER.CODE8 = model_options()
    return cfg
