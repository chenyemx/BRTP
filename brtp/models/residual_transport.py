"""BRTP source-anchor residual transport and registered ablation controls.

Kernel ridge regression, frozen text anchors and residual tuning are prior art.
See THIRD_PARTY.md for source attribution.
No target labels, target images or fitted target statistics enter this module.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from brtp.config import model_options, validate_template, validated_model_options
from brtp.models.shared_prompt import SharedPromptCLIP


def tangent(delta, anchor):
    """Remove radial movement relative to a unit text anchor."""
    return delta - (delta * anchor).sum(-1, keepdim=True) * anchor


def bound_residual(delta, radius):
    norm = delta.norm(dim=-1, keepdim=True)
    return delta * (radius / norm.clamp_min(1e-8)).clamp(max=1.0)


class AnchorKernel(nn.Module):
    """Frozen RBF geometry and exact leave-one-donor-out ridge operators.

    Notation: bandwidth is b, ridge is rho, and inverse is A.
    Coverage g_q = k_q (K + rho I)^-1 k_q.T is algebraic support coverage,
    not a calibrated probability. lambda is reserved for the auxiliary loss.
    A query never changes the source kernel/solve.
    """

    def __init__(self, anchors, bandwidth=10.0, ridge=0.1, *, wrong_donor=False):
        super().__init__()
        if anchors.ndim != 2 or anchors.shape[0] < 2:
            raise ValueError("At least two source class anchors are required")
        if not (math.isfinite(bandwidth) and bandwidth > 0 and math.isfinite(ridge) and ridge > 0):
            raise ValueError("Positive finite kernel bandwidth/ridge required")
        if not bool(torch.isfinite(anchors).all()) or bool((anchors.norm(dim=-1) < 1e-8).any()):
            raise ValueError("Invalid source anchors")
        self.bandwidth, self.ridge = float(bandwidth), float(ridge)
        with torch.no_grad(), autocast(enabled=False):
            a = F.normalize(anchors.detach().float(), dim=-1)
            k = self.kernel(a, a)
            eye = torch.eye(len(a), device=a.device)
            inverse = torch.linalg.solve(k + self.ridge * eye, eye)
            loo = torch.zeros_like(k)
            loo_coverage = torch.zeros(len(a), device=a.device)
            for i in range(len(a)):
                mask = torch.arange(len(a), device=a.device) != i
                sub = k[mask][:, mask] + self.ridge * eye[mask][:, mask]
                weights = torch.linalg.solve(sub, k[i, mask])
                loo[i, mask] = weights
                loo_coverage[i] = (weights * k[i, mask]).sum().clamp(0, 1)
        # Control 19: same C-1 donor count, exclude the next frozen source class.
        # This is a genuine reduced solve, not zeroing a full-system coefficient.
        wrong_operator = None
        excluded = None
        if wrong_donor:
            with torch.no_grad(), autocast(enabled=False):
                excluded = (torch.arange(len(a), device=a.device) + 1) % len(a)
                wrong_operator = torch.zeros_like(k)
                for i in range(len(a)):
                    mask = torch.arange(len(a), device=a.device) != excluded[i]
                    sub = k[mask][:, mask] + self.ridge * eye[mask][:, mask]
                    wrong_operator[i, mask] = torch.linalg.solve(sub, k[i, mask])
        self.register_buffer("wrong_donor", wrong_operator, persistent=False)
        self.register_buffer("wrong_excluded_indices", excluded, persistent=False)
        self.register_buffer("anchors", a, persistent=False)
        self.register_buffer("inverse", inverse, persistent=False)
        self.register_buffer("loo", loo, persistent=False)
        self.register_buffer("loo_coverage", loo_coverage, persistent=False)

    def kernel(self, left, right):
        return torch.exp(-self.bandwidth * (1 - left @ right.t()).clamp_min(0))

    @torch.no_grad()
    def query(self, anchors):
        with autocast(enabled=False):
            k = self.kernel(F.normalize(anchors.detach().float(), dim=-1), self.anchors)
            weights = k @ self.inverse
            coverage = (weights * k).sum(-1, keepdim=True).clamp(0, 1)
        return weights, coverage


class ResidualTransportCLIP(SharedPromptCLIP):
    """A frozen zero-shot anchor plus a bounded, transferable prompt update.

    Delta is p_Theta(c)-p_Theta_0(c), so the initial classifier is exactly
    the frozen teacher ensemble. Source vocabulary is restored from the source
    checkpoint contract. Persistent initial_ctx belongs to the checkpoint.
    The tangent-projected delta is d_c, source_delta stacks D,
    coverage is g_q, the blended residual is r_q, and _compose returns h_q.
    MAX_RESIDUAL_NORM is the bound R.
    """

    def __init__(self, backbone, *args, options=None, code8=None, **kwargs):
        if options is not None and code8 is not None:
            raise ValueError("Supply options once (code8 is a historical keyword alias)")
        options = options if options is not None else code8
        options = validated_model_options(model_options() if options is None else options)
        super().__init__(backbone, *args, **kwargs)
        self.options = options
        self._inference_bound_override = None
        self.templates = tuple(options.TEMPLATES)
        self.prompt_learner.register_buffer("initial_ctx", self.prompt_learner.ctx.detach().clone())
        self.source_names = ()
        self.geometry = None
        self._initial_features = {}
        self._last_source_delta = None
        self.last_diagnostics = None

    def clear_derived_cache(self):
        super().clear_derived_cache()
        self._initial_features.clear()
        self._last_source_delta = None
        self.last_diagnostics = None

    @torch.no_grad()
    def encode_teacher(self, texts, templates=()):
        """Fixed FP32 anchors used at both training and inference.

        The fixed ensemble is used in the classifier. The first AMP call must
        not determine cached anchor precision or template grouping.
        """
        from clip import clip

        if (
            isinstance(texts, str)
            or not texts
            or not all(isinstance(t, str) and t.strip() for t in texts)
        ):
            raise ValueError("Nonempty class-name sequence required")
        templates = tuple(templates or (self.teacher_template,))
        for template in templates:
            validate_template(template)
        device = self.prompt_learner.ctx.device
        key = ("brtp_fp32", tuple(texts), templates, str(device))
        if key not in self._frozen_cache:
            tokens = clip.tokenize([name.replace("_", " ") for name in texts], truncate=False)
            if len({tuple(row) for row in tokens.tolist()}) != len(texts):
                raise ValueError("Native class names collide after CLIP tokenization")
            rendered = [t.format(name.replace("_", " ")) for t in templates for name in texts]
            with autocast(enabled=False):
                ids = clip.tokenize(rendered, truncate=False).to(device)
                features = F.normalize(self.backbone.encode_text(ids).float(), dim=-1)
                average = features.reshape(len(templates), len(texts), -1).mean(0)
                if not bool(torch.isfinite(average).all()) or bool(
                    (average.norm(dim=-1) < 1e-8).any()
                ):
                    raise ValueError("Invalid frozen ensemble anchor")
                self._frozen_cache[key] = F.normalize(average, dim=-1).detach()
        return self._frozen_cache[key]

    @torch.no_grad()
    def configure_source(self, names):
        if (
            isinstance(names, str)
            or len(names) < 2
            or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n.strip() for n in names)
        ):
            raise ValueError("Unique nonempty source class names are required")
        names = tuple(names)
        # Commit the bank only after encoding and the kernel solve succeed.
        # Otherwise an exception could leave old geometry under new names.
        cached_anchors = self._frozen_cache.copy()
        try:
            z = self.encode_teacher(names, self.templates)
            geometry = AnchorKernel(
                z,
                self.options.BANDWIDTH,
                self.options.RIDGE,
                wrong_donor=self.options.AUXILIARY_WRONG_DONOR,
            )
        except Exception:
            self._frozen_cache.clear()
            self._frozen_cache.update(cached_anchors)
            raise
        self.source_names = names
        self.geometry = geometry
        self.clear_derived_cache()

    @torch.no_grad()
    def initial_features(self, texts):
        key = (tuple(texts), str(self.prompt_learner.ctx.device), bool(torch.is_autocast_enabled()))
        if key not in self._initial_features:
            # Encode the fixed context without swapping parameters or mutating
            # the trainable context; cache only the detached initial features.
            self._initial_features[key] = self.encode_with_context(
                texts, self.prompt_learner.initial_ctx
            ).detach()
        return self._initial_features[key]

    def source_delta(self):
        prompted = super().encode_texts(self.source_names)
        initial = self.initial_features(self.source_names)
        return self._project(prompted.float() - initial.float(), self.geometry.anchors)

    def _project(self, delta, anchor):
        # Used for direct updates, source updates and every composition (including DEC).
        return tangent(delta, anchor) if self.options.TANGENT_PROJECTION else delta

    def effective_bound(self):
        override = self._inference_bound_override
        if override is not None:
            if self.training:
                raise RuntimeError("Inference-only bound override cannot be used in training mode")
            return override
        return bool(self.options.RESIDUAL_BOUND), float(self.options.MAX_RESIDUAL_NORM)

    def _compose(self, anchor, residual):
        residual = self._project(residual.float(), anchor.float())
        enabled, radius = self.effective_bound()
        if enabled:
            residual = bound_residual(residual, radius)
        return F.normalize(anchor.float() + residual, dim=-1)

    def encode_texts(self, texts, zero_shot=False, text_kind="native"):
        if zero_shot:
            if text_kind != "native":
                return super().encode_texts(texts, True, text_kind)
            return self.encode_teacher(texts, self.templates)
        if text_kind != "native":
            raise ValueError("BRTP accepts native class names only")
        if self.geometry is None:
            raise RuntimeError(
                "Source anchors must be configured from training or a source checkpoint"
            )
        if isinstance(texts, str):
            raise ValueError("Nonempty class-name sequence required")
        names = tuple(texts)
        anchor = self.encode_teacher(names, self.templates)
        prompted = super().encode_texts(names)
        delta = self._project(prompted.float() - self.initial_features(names).float(), anchor)
        source_delta = delta if names == self.source_names else self.source_delta()
        if names == self.source_names:
            self._last_source_delta = source_delta
        # Disable autocast for the small geometry operations, including inference.
        with autocast(enabled=False):
            weights, coverage = self.geometry.query(anchor)
            if self.options.COVERAGE_MODE == "fixed_half":
                coverage = torch.full_like(coverage, 0.5)
            if not self.options.COVERAGE:
                coverage = torch.ones_like(coverage)
            residual = coverage * delta
            if self.options.TRANSPORT:
                residual = residual + (1 - coverage) * (weights @ source_delta.float())
            projected = self._project(residual.float(), anchor.float())
            bound_enabled, bound_radius = self.effective_bound()
            residual_norm = projected.detach().norm(dim=-1)
            above_radius = float((residual_norm > bound_radius).float().mean())
            self.last_diagnostics = dict(
                candidate_names=list(names),
                coverage=coverage.detach()[:, 0].cpu().tolist(),
                pre_bound_residual_norm=residual_norm.cpu().tolist(),
                bound_active_fraction=above_radius,
                source_donor_count=len(self.source_names),
                tangent_projection=bool(self.options.TANGENT_PROJECTION),
                effective_bound_enabled=bound_enabled,
                effective_bound_radius=bound_radius,
                actually_clipped_class_fraction=above_radius if bound_enabled else 0.0,
                inference_bound_override=self._inference_bound_override is not None,
            )
            return self._compose(anchor, residual)

    def crossfit_text_features(self):
        if self._last_source_delta is None:
            raise RuntimeError("Encode source candidates before the donor-exclusion objective")
        with autocast(enabled=False):
            operator = self.geometry.loo
            if self.options.AUXILIARY_SELF_DONOR:
                operator = (
                    self.geometry.kernel(self.geometry.anchors, self.geometry.anchors)
                    @ self.geometry.inverse
                )
            if self.options.AUXILIARY_WRONG_DONOR:
                if self.geometry.wrong_donor is None:
                    raise RuntimeError("Wrong-exclusion geometry was not configured")
                operator = self.geometry.wrong_donor
            residual = operator @ self._last_source_delta.float()
            return self._compose(self.geometry.anchors, residual)

    @torch.no_grad()
    def geometry_audit(self):
        _, coverage = self.geometry.query(self.geometry.anchors)
        return dict(
            source_names=list(self.source_names),
            templates=list(self.templates),
            bandwidth=self.options.BANDWIDTH,
            ridge=self.options.RIDGE,
            max_residual_norm=self.options.MAX_RESIDUAL_NORM,
            source_coverage=coverage[:, 0].cpu().tolist(),
            leave_one_donor_out_coverage=self.geometry.loo_coverage.cpu().tolist(),
            leave_one_donor_out_diagonal=self.geometry.loo.diag().cpu().tolist(),
            tangent_projection=bool(self.options.TANGENT_PROJECTION),
            bound_enabled=bool(self.options.RESIDUAL_BOUND),
            auxiliary_wrong_donor=bool(self.options.AUXILIARY_WRONG_DONOR),
            wrong_donor_rule="exclude source index (i+1) modulo C; keep self; solve C-1 system"
            if self.options.AUXILIARY_WRONG_DONOR
            else None,
            wrong_excluded_indices=self.geometry.wrong_excluded_indices.cpu().tolist()
            if self.geometry.wrong_excluded_indices is not None
            else None,
            wrong_excluded_names=[
                self.source_names[int(i)] for i in self.geometry.wrong_excluded_indices
            ]
            if self.geometry.wrong_excluded_indices is not None
            else None,
            geometry_scope="source_training_class_names_only",
            coverage_interpretation="kernel algebraic coverage; not a calibrated probability",
            training_scope_note="Donor exclusion is an auxiliary objective, not held-out-class validation",
        )
