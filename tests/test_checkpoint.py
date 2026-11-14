"""Native checkpoint and update invariants using tiny random CPU models."""

import copy
import io
import unittest

import torch
from torch.nn import functional as F
from brtp.models.adapter import build_model, training_loss
from clip import clip
from test_mechanisms import tiny


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_saved_prompt_restores_initial_context_and_native_source_order(self):
        names = ["forest", "bridge", "airport"]
        backbone = tiny()
        model = build_model(copy.deepcopy(backbone), {}, names)
        generator = torch.Generator().manual_seed(53)
        with torch.no_grad():
            model.prompt_learner.ctx.add_(
                torch.randn(model.prompt_learner.ctx.shape, generator=generator) * 0.2
            )
        state = model.prompt_learner.state_dict()
        self.assertEqual(set(state), {"ctx", "initial_ctx"})
        archive = io.BytesIO()
        torch.save(state, archive)
        archive.seek(0)
        saved = torch.load(archive, map_location="cpu", weights_only=False)
        restored = build_model(copy.deepcopy(backbone), {}, names, prompt_state=saved)
        self.assertEqual(restored.source_names, tuple(names))
        for candidates in (names, ["river", "residential area"]):
            self.assertTrue(
                torch.equal(model.encode_texts(candidates), restored.encode_texts(candidates))
            )
        with self.assertRaises(RuntimeError):
            build_model(copy.deepcopy(backbone), {}, names, prompt_state={"ctx": saved["ctx"]})

    def test_text_initialization_uses_frozen_prefix_for_every_context_length(self):
        sentence = "a photo of a satellite image of a remote sensing scene captured from above at high resolution"
        backbone = tiny()
        ids = clip.tokenize(sentence, truncate=False)
        with torch.no_grad():
            embeddings = backbone.token_embedding(ids)[0]
        for n_ctx in (1, 4, 8, 12, 16):
            with self.subTest(n_ctx=n_ctx):
                model = build_model(
                    copy.deepcopy(backbone),
                    {"n_ctx": n_ctx, "context_init": sentence},
                    ["forest", "airport"],
                )
                self.assertTrue(torch.equal(model.prompt_learner.ctx, embeddings[1 : 1 + n_ctx]))
                self.assertTrue(
                    torch.equal(model.prompt_learner.initial_ctx, embeddings[1 : 1 + n_ctx])
                )

    def test_optimizer_changes_only_context_and_freezes_initial_context(self):
        model = build_model(tiny(), {}, ["forest", "airport", "bridge"])
        initial = model.prompt_learner.initial_ctx.clone()
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.SGD(
            model.prompt_learner.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005
        )
        features = F.normalize(
            torch.randn(6, 32, generator=torch.Generator().manual_seed(17)), dim=-1
        )
        loss, pieces = training_loss(
            model, features, model.source_names, torch.tensor([0, 1, 2, 0, 1, 2]), 0.25
        )
        self.assertTrue(torch.equal(loss, pieces["ce"] + pieces["donor_weighted"]))
        loss.backward()
        optimizer.step()
        changed = [k for k, v in model.named_parameters() if not torch.equal(before[k], v)]
        self.assertEqual(changed, ["prompt_learner.ctx"])
        self.assertTrue(torch.equal(initial, model.prompt_learner.initial_ctx))
        self.assertTrue(all(v.grad is None for v in model.backbone.parameters()))

    def test_resume_preserves_momentum_and_cosine_schedule_without_warmup(self):
        def pair():
            parameter = torch.nn.Parameter(torch.tensor([0.1, -0.3]))
            optimizer = torch.optim.SGD([parameter], lr=0.002, momentum=0.9, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            return parameter, optimizer, scheduler

        param, optimizer, scheduler = pair()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.002)
        for epoch in range(28):
            param.grad = torch.tensor([0.2, -0.4])
            optimizer.step()
            scheduler.step()
        resumed, other, other_scheduler = pair()
        with torch.no_grad():
            resumed.copy_(param)
        other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        other_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        for epoch in range(28, 50):
            param.grad = torch.tensor([0.2, -0.4])
            resumed.grad = param.grad.clone()
            optimizer.step()
            scheduler.step()
            other.step()
            other_scheduler.step()
            self.assertTrue(torch.equal(param, resumed))
            self.assertEqual(optimizer.param_groups[0]["lr"], other.param_groups[0]["lr"])
        self.assertEqual(scheduler.last_epoch, 50)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
