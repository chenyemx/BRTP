"""Independent mechanism, gradient and checkpoint invariants; synthetic CPU fixtures only."""

import copy
import unittest
from unittest import mock
import numpy as np
import torch
from torch.nn import functional as F
from yacs.config import CfgNode as CN
from brtp.models.shared_prompt import SharedPromptCLIP
from brtp.models.adapter import build_model, inference_bound_override, model_contract, training_loss
from brtp.models.residual_transport import AnchorKernel, ResidualTransportCLIP, tangent
from clip.model import CLIP


def tiny():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        return (
            CLIP(
                embed_dim=32,
                image_resolution=16,
                vision_layers=1,
                vision_width=64,
                vision_patch_size=8,
                context_length=77,
                vocab_size=49408,
                transformer_width=32,
                transformer_heads=4,
                transformer_layers=1,
            )
            .float()
            .eval()
        )


def options(**updates):
    o = dict(
        TEMPLATES=["a photo of a {}.", "a satellite image of a {}.", "an aerial image of a {}."],
        BANDWIDTH=10.0,
        RIDGE=0.1,
        MAX_RESIDUAL_NORM=0.25,
        TRANSPORT=True,
        COVERAGE=True,
        RESIDUAL_BOUND=True,
        COVERAGE_MODE="adaptive",
        AUXILIARY_SELF_DONOR=False,
        TANGENT_PROJECTION=True,
        AUXILIARY_WRONG_DONOR=False,
    )
    o.update(updates)
    return CN(o)


def fixture(opts=None, cls=ResidualTransportCLIP, backbone=None):
    torch.manual_seed(2)
    m = cls(
        tiny() if backbone is None else backbone,
        n_ctx=4,
        ctx_init="a photo of a",
        renderer_version="native_query_v1",
        native_template="{}.",
        code8=opts or options(),
    )
    m.configure_source(["forest", "bridge", "airport"])
    return m


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_projection_switch_controls_direct_source_final_and_auxiliary_paths(self):
        for project in [False, True]:
            for bounded in [False, True]:
                with self.subTest(project=project, bounded=bounded):
                    m = fixture(options(TANGENT_PROJECTION=project, RESIDUAL_BOUND=bounded))
                    names = m.source_names
                    target = ("river", "stadium")
                    torch.manual_seed(12)
                    sa = F.normalize(torch.randn(3, 32), dim=-1)
                    ta = F.normalize(torch.randn(2, 32), dim=-1)
                    si = F.normalize(torch.randn(3, 32), dim=-1)
                    ti = F.normalize(torch.randn(2, 32), dim=-1)
                    sp = F.normalize(torch.randn(3, 32), dim=-1)
                    tp = F.normalize(torch.randn(2, 32), dim=-1)
                    m.geometry = AnchorKernel(sa, 10, 0.1)
                    sa = m.geometry.anchors
                    m.encode_teacher = lambda n, templates=(): sa if tuple(n) == names else ta
                    m.initial_features = lambda n: si if tuple(n) == names else ti

                    def parent_encode(self, n, *args, **kwargs):
                        return sp if tuple(n) == names else tp

                    def proj(d, a):
                        return d - (d * a).sum(-1, keepdim=True) * a if project else d

                    def compose(a, d):
                        d = proj(d, a)
                        if bounded:
                            d = d * (0.25 / d.norm(dim=-1, keepdim=True).clamp_min(1e-08)).clamp(
                                max=1.0
                            )
                        return F.normalize(a + d, dim=-1)

                    with mock.patch.object(SharedPromptCLIP, "encode_texts", parent_encode):
                        sd = proj(sp - si, sa)
                        self.assertTrue(torch.equal(m.source_delta(), sd))
                        m.encode_texts(names)
                        self.assertTrue(torch.equal(m._last_source_delta, sd))
                        w, g = m.geometry.query(ta)
                        delta = proj(tp - ti, ta)
                        expected = compose(ta, g * delta + (1 - g) * (w @ sd))
                        self.assertTrue(torch.equal(m.encode_texts(target), expected))
                        self.assertTrue(
                            torch.equal(
                                m.crossfit_text_features(), compose(sa, m.geometry.loo @ sd)
                            )
                        )
                    if not project:
                        self.assertGreater(float((sd * sa).sum(-1).abs().max()), 0.01)

    def test_wrong_exclusion_matches_independent_reduced_solve_not_zeroed_full_system(self):
        a = torch.tensor([[1.0, 0.0, 0.0], [0.95, 0.3, 0.0], [0.8, 0.5, 0.2], [0.7, 0.4, 0.5]])
        a = F.normalize(a, dim=-1)
        g = AnchorKernel(a, 2.0, 0.1, wrong_donor=True)
        aa = a.double().numpy()
        k = np.exp(-2.0 * np.maximum(1 - aa @ aa.T, 0.0))
        expected = np.zeros((4, 4))
        for i in range(4):
            excluded = (i + 1) % 4
            mask = np.arange(4) != excluded
            expected[i, mask] = np.linalg.solve(k[np.ix_(mask, mask)] + 0.1 * np.eye(3), k[i, mask])
            self.assertEqual(float(g.wrong_donor[i, excluded]), 0.0)
            self.assertGreater(float(g.wrong_donor[i, i]), 0.0)
        self.assertEqual(g.wrong_excluded_indices.tolist(), [1, 2, 3, 0])
        self.assertTrue(np.allclose(g.wrong_donor.numpy(), expected, atol=3e-06))
        naive = g.kernel(g.anchors, g.anchors) @ g.inverse
        naive[torch.arange(4), g.wrong_excluded_indices] = 0
        self.assertGreater(float((g.wrong_donor - naive).abs().max()), 0.001)
        self.assertFalse(torch.equal(g.wrong_donor, g.loo))

    def test_wrong_exclusion_with_two_classes_retains_self(self):
        g = AnchorKernel(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 10.0, 0.1, wrong_donor=True)
        self.assertTrue(torch.allclose(g.wrong_donor, torch.eye(2) / 1.1))
        self.assertEqual(g.wrong_excluded_indices.tolist(), [1, 0])

    def test_wrong_exclusion_is_used_by_aux_loss_and_has_context_gradients(self):
        m = fixture(options(AUXILIARY_WRONG_DONOR=True))
        with torch.no_grad():
            m.prompt_learner.ctx.add_(torch.randn_like(m.prompt_learner.ctx) * 0.2)
        m.encode_texts(m.source_names)
        actual = m.crossfit_text_features()
        expected = m._compose(m.geometry.anchors, m.geometry.wrong_donor @ m._last_source_delta)
        self.assertTrue(torch.equal(actual, expected))
        v = F.normalize(torch.randn(4, 32), dim=-1)
        y = torch.tensor([0, 1, 2, 1])
        F.cross_entropy(20 * v @ actual.T, y).backward()
        self.assertTrue(torch.isfinite(m.prompt_learner.ctx.grad).all())
        self.assertGreater(float(m.prompt_learner.ctx.grad.abs().sum()), 0.0)
        self.assertTrue(all((p.grad is None for p in m.backbone.parameters())))
        audit = m.geometry_audit()
        self.assertEqual(audit["wrong_excluded_names"], ["bridge", "airport", "forest"])

    def test_wrong_and_self_donor_cannot_be_enabled_together(self):
        with self.assertRaises(ValueError):
            fixture(options(AUXILIARY_SELF_DONOR=True, AUXILIARY_WRONG_DONOR=True))

    def test_inference_override_is_reversible_and_never_mutates_training_contract(self):
        m = fixture()
        m.eval()
        a = m.geometry.anchors
        d = tangent(torch.randn_like(a) * 5, a)
        before = m._compose(a, d)
        state = {k: v.clone() for k, v in m.prompt_learner.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "sentinel"):
            with inference_bound_override(m, False):
                self.assertFalse(torch.allclose(m._compose(a, d), before))
                self.assertTrue(m.options.RESIDUAL_BOUND)
                self.assertFalse(model_contract(m)["effective_bound_enabled"])
                with inference_bound_override(m, True, 0.125):
                    self.assertEqual(m.effective_bound(), (True, 0.125))
                self.assertEqual(m.effective_bound(), (False, 0.25))
                raise ValueError("sentinel")
        self.assertIsNone(m._inference_bound_override)
        self.assertTrue(torch.equal(m._compose(a, d), before))
        self.assertTrue(
            all((torch.equal(v, state[k]) for k, v in m.prompt_learner.state_dict().items()))
        )
        m.train()
        with self.assertRaises(RuntimeError):
            with inference_bound_override(m, False):
                pass
        m.eval()
        with inference_bound_override(m, False):
            m.train()
            with self.assertRaises(RuntimeError):
                m._compose(a, d)

    def test_pure_coop_is_identical_to_shared_prompt_and_does_not_touch_geometry(self):
        b = tiny()
        with mock.patch.object(
            AnchorKernel, "__init__", side_effect=AssertionError("No geometry allowed")
        ):
            pure = build_model(
                copy.deepcopy(b),
                {"classifier": "CoOp", "lambda_weight": 0},
                ["forest", "bridge", "airport"],
            )
        expected = SharedPromptCLIP(
            copy.deepcopy(b),
            n_ctx=4,
            ctx_init="a photo of a",
            renderer_version="native_query_v1",
            native_template="{}.",
        )
        with torch.no_grad():
            delta = torch.randn_like(pure.prompt_learner.ctx) * 0.2
            pure.prompt_learner.ctx.add_(delta)
            expected.prompt_learner.ctx.add_(delta)
        self.assertFalse(hasattr(pure, "geometry"))
        self.assertFalse(hasattr(pure, "options"))
        with mock.patch.object(
            pure,
            "encode_teacher",
            side_effect=AssertionError("No frozen-anchor branch allowed"),
            create=True,
        ):
            for names in [pure.source_names, ["river", "stadium"]]:
                self.assertTrue(torch.equal(pure.encode_texts(names), expected.encode_texts(names)))
            v = F.normalize(torch.randn(4, 32), dim=-1)
            y = torch.tensor([0, 1, 2, 0])
            loss, _ = training_loss(pure, v, pure.source_names, y, 0.0)
            ref = F.cross_entropy(
                expected.backbone.logit_scale.exp()
                * v
                @ expected.encode_texts(pure.source_names).T,
                y,
            )
            self.assertTrue(torch.equal(loss, ref))
            loss.backward()
            ref.backward()
            self.assertTrue(
                torch.equal(pure.prompt_learner.ctx.grad, expected.prompt_learner.ctx.grad)
            )
        with self.assertRaises(ValueError):
            training_loss(pure, v, pure.source_names, y, 0.25)
