"""Checkpoint reuse and class ordering must preserve the declared classifier."""

import copy
import unittest

import torch
from torch.nn import functional as F

from brtp.models.adapter import build_model, training_loss
from test_mechanisms import tiny


class ModelReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.backbone = tiny()
        cls.names = ("forest", "bridge", "airport")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_prompt_checkpoint_reuse_matches_fresh_model_with_new_initial_context(self):
        model = build_model(copy.deepcopy(self.backbone), {}, self.names)
        model.encode_texts(self.names)
        model.encode_texts(("river", "stadium"))
        state = copy.deepcopy(model.prompt_learner.state_dict())
        rng = torch.Generator().manual_seed(307)
        state["ctx"] += torch.randn(state["ctx"].shape, generator=rng) * 0.2
        state["initial_ctx"] += torch.randn(state["initial_ctx"].shape, generator=rng) * 0.2
        model.prompt_learner.load_state_dict(state, strict=True)
        fresh = build_model(copy.deepcopy(self.backbone), {}, self.names, prompt_state=state)
        for candidates in (self.names, ("river", "stadium")):
            with self.subTest(candidates=candidates):
                self.assertTrue(
                    torch.equal(model.encode_texts(candidates), fresh.encode_texts(candidates))
                )

    def test_full_state_load_and_tensor_relocation_discard_derived_state(self):
        model = build_model(copy.deepcopy(self.backbone), {}, self.names)
        model.encode_texts(self.names)
        model.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(model._initial_features, {})
        self.assertIsNone(model._last_source_delta)
        before = model.encode_texts(self.names)
        model.to("cpu")
        self.assertEqual(model._initial_features, {})
        self.assertIsNone(model._last_source_delta)
        self.assertTrue(torch.equal(before, model.encode_texts(self.names)))

    def test_donor_loss_rejects_reordered_candidates_before_using_old_source_graph(self):
        model = build_model(copy.deepcopy(self.backbone), {}, self.names)
        model.encode_texts(self.names)
        visual = F.normalize(
            torch.randn(3, 32, generator=torch.Generator().manual_seed(43)), dim=-1
        )
        labels = torch.tensor([0, 1, 2])
        for names in (self.names[::-1], ("river", "stadium", "church")):
            with self.subTest(names=names), self.assertRaisesRegex(
                ValueError, "source-class order"
            ):
                training_loss(model, visual, names, labels, 0.25)
        loss, _ = training_loss(model, visual, self.names, labels, 0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(model.prompt_learner.ctx.grad).all())


if __name__ == "__main__":
    unittest.main()
