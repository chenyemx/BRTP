"""Model construction and native classification for retained BRTP ablations.

The caller owns data, optimizer, scheduler, seeding and artifact contracts.
No weights are downloaded, jobs launched or datasets discovered here.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from collections.abc import Mapping, Sequence

from brtp.config import model_options
from brtp.models.shared_prompt import SharedPromptCLIP
from .residual_transport import ResidualTransportCLIP


class PureCoOp(SharedPromptCLIP):
    """Shared context CE classifier; no BRTP residual or geometry computation."""

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, **kwargs)
        # Bookkeeping only: forward is inherited unchanged and never subtracts it.
        self.prompt_learner.register_buffer("initial_ctx", self.prompt_learner.ctx.detach().clone())
        self.source_names = ()
        self.last_diagnostics = None

    def configure_source(self, names: Sequence[str]) -> None:
        if isinstance(names, str) or len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Unique source class names are required")
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("Source class names must be nonempty strings")
        self.source_names = tuple(names)
        self.clear_derived_cache()

    def crossfit_text_features(self):
        raise RuntimeError("Pure CoOp has no donor-exclusion objective")

    def geometry_audit(self) -> dict:
        return {
            "classifier": "pure_CoOp",
            "BRTP_geometry": False,
            "source_names": list(self.source_names),
            "initial_ctx_buffer_role": "provenance only; unused in prediction",
        }


def build_model(
    backbone,
    parameters: Mapping,
    source_names: Sequence[str],
    *,
    prompt_state: Mapping | None = None,
    device=None,
    ctx_init: str = "a photo of a",
):
    """Return a model with one trainable shared context and frozen backbone.

    Parameters follow the archived recipe names. The supplied source-name
    order defines the cyclic wrong-donor exclusion.
    This function does not reseed: call the frozen runner's seed_all first.
    """
    for key in ("transport", "bound_enabled", "self_donor", "tangent_projection", "wrong_donor"):
        if key in parameters and not isinstance(parameters[key], bool):
            raise TypeError(key + " must be a JSON/Python boolean")
    classifier = parameters.get("classifier", "BRTP")
    n_ctx = parameters.get("n_ctx", 4)
    ctx_init = parameters.get("context_init", ctx_init)
    common = dict(
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        renderer_version="native_query_v1",
        native_template="{}.",
    )
    if classifier in ("CoOp", "pure_CoOp", "CoOp_matched_recipe"):
        if float(parameters.get("lambda_weight", 0.0)) != 0.0:
            raise ValueError("Pure CoOp requires lambda_weight=0")
        model = PureCoOp(backbone, **common)
    elif classifier == "BRTP":
        opts = model_options()
        for parameter, option in {
            "templates": "TEMPLATES",
            "bandwidth": "BANDWIDTH",
            "ridge": "RIDGE",
            "radius": "MAX_RESIDUAL_NORM",
            "transport": "TRANSPORT",
            "bound_enabled": "RESIDUAL_BOUND",
            "coverage_mode": "COVERAGE_MODE",
            "self_donor": "AUXILIARY_SELF_DONOR",
            "tangent_projection": "TANGENT_PROJECTION",
            "wrong_donor": "AUXILIARY_WRONG_DONOR",
        }.items():
            if parameter in parameters:
                setattr(opts, option, parameters[parameter])
        model = ResidualTransportCLIP(backbone, options=opts, **common)
    else:
        raise ValueError("Unknown classifier: " + str(classifier))
    if device is not None:
        model = model.to(device)
    if prompt_state is not None:
        model.prompt_learner.load_state_dict(prompt_state, strict=True)
    model.configure_source(source_names)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if names != ["prompt_learner.ctx"]:
        raise ValueError("Only prompt_learner.ctx may be trainable: " + repr(names))
    if model.prompt_learner.ctx.shape[0] != n_ctx:
        raise ValueError("Expected %d shared context tokens" % n_ctx)
    return model


@contextmanager
def inference_bound_override(model, enabled: bool, radius: float | None = None):
    """Option 11: temporarily change only clipping during evaluation.

    Neither options nor checkpoint tensors are mutated. Nested overrides are
    restored even on failure. Training mode is rejected at entry and at use.
    Record enabled/radius and source checkpoint SHA in each evaluation contract.
    """
    if not isinstance(model, ResidualTransportCLIP):
        raise TypeError("A bound override is meaningful only for BRTP")
    if model.training:
        raise RuntimeError("Call model.eval() before inference-only intervention")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    actual_radius = float(model.options.MAX_RESIDUAL_NORM if radius is None else radius)
    if not math.isfinite(actual_radius) or actual_radius <= 0:
        raise ValueError("A positive finite radius is required")
    previous = model._inference_bound_override
    model._inference_bound_override = (enabled, actual_radius)
    try:
        yield model
    finally:
        model._inference_bound_override = previous


def training_loss(model, visual, names: Sequence[str], labels, lambda_weight: float):
    """The historical CE/DEC loss order, with a fail-closed pure-CoOp branch."""
    if isinstance(model, PureCoOp) and float(lambda_weight) != 0.0:
        raise ValueError("Pure CoOp cannot have an auxiliary BRTP loss")
    if lambda_weight and (isinstance(names, str) or tuple(names) != model.source_names):
        raise ValueError("Donor CE requires candidates in the configured source-class order")
    from brtp.objective import classification_objective

    text = model.encode_texts(names)
    scale = model.backbone.logit_scale.exp()
    logits = scale * visual @ text.t()
    donor = scale * visual @ model.crossfit_text_features().t() if lambda_weight else None
    return classification_objective(logits, labels, donor, lambda_weight)


def model_contract(model) -> dict:
    """Describe the declared recipe, not initialization inferred from weights."""
    initialization = model.prompt_learner.ctx_init
    common = {
        "n_ctx": model.prompt_learner.ctx.shape[0],
        "initialization": initialization,
        "initialization_mode": "text_prefix" if initialization else "normal_std_0.02",
        "trainable_parameters": model.prompt_learner.ctx.numel(),
    }
    if isinstance(model, PureCoOp):
        return {
            **common,
            "classifier": "pure_CoOp",
            "prediction": "normalized learned text features",
            "loss": "CE only",
            "g": None,
            "BPR": False,
            "SRT": False,
            "DEC": False,
        }
    enabled, radius = model.effective_bound()
    return {
        **common,
        "classifier": "BRTP",
        "tangent_projection": bool(model.options.TANGENT_PROJECTION),
        "training_bound_enabled": bool(model.options.RESIDUAL_BOUND),
        "effective_bound_enabled": enabled,
        "effective_bound_radius": radius,
        "inference_only_bound_override": model._inference_bound_override is not None,
        "wrong_donor": bool(model.options.AUXILIARY_WRONG_DONOR),
        "self_donor": bool(model.options.AUXILIARY_SELF_DONOR),
        "source_names": list(model.source_names),
        "wrong_donor_exclusion": (
            "(i+1)%C in frozen source-name order" if model.options.AUXILIARY_WRONG_DONOR else None
        ),
    }
