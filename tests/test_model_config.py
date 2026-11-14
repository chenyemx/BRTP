"""Configuration errors must not silently change a reproducibility recipe."""

import copy
import unittest

import torch

from brtp.config import model_options, validated_model_options
from brtp.models.adapter import build_model, model_contract
from brtp.models.residual_transport import ResidualTransportCLIP
from brtp.models.shared_prompt import PromptLearner, SharedPromptCLIP
from test_mechanisms import tiny


class ModelConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.backbone = tiny()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_contract_records_text_or_random_recipe_without_new_checkpoint_keys(self):
        sentence = "a photo of a satellite image of a remote sensing scene"
        for classifier in ("BRTP", "CoOp"):
            for initialization in ("a photo of a", sentence, ""):
                with self.subTest(classifier=classifier, initialization=initialization):
                    model = build_model(
                        copy.deepcopy(self.backbone),
                        {"classifier": classifier, "context_init": initialization},
                        ["forest", "bridge"],
                    )
                    contract = model_contract(model)
                    self.assertEqual(contract["initialization"], initialization)
                    self.assertEqual(
                        contract["initialization_mode"],
                        "text_prefix" if initialization else "normal_std_0.02",
                    )
                    self.assertEqual(set(model.prompt_learner.state_dict()), {"ctx", "initial_ctx"})

    def test_context_length_is_not_silently_truncated_or_coerced(self):
        for n_ctx in (True, 4.5, "4", 0, 33):
            with self.subTest(n_ctx=n_ctx):
                state = torch.get_rng_state().clone()
                with self.assertRaises((TypeError, ValueError)):
                    build_model(
                        copy.deepcopy(self.backbone), {"n_ctx": n_ctx}, ["forest", "bridge"]
                    )
                self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_context_initialization_requires_explicit_string(self):
        for initialization in (None, False, 0, ["a photo of a"]):
            with self.subTest(initialization=initialization):
                state = torch.get_rng_state().clone()
                with self.assertRaises(TypeError):
                    PromptLearner(self.backbone, ctx_init=initialization)
                self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_direct_model_rejects_ambiguous_booleans_and_invalid_geometry(self):
        cases = [
            ("TRANSPORT", "false"),
            ("COVERAGE", 1),
            ("RESIDUAL_BOUND", "true"),
            ("AUXILIARY_SELF_DONOR", 0),
            ("TANGENT_PROJECTION", "false"),
            ("AUXILIARY_WRONG_DONOR", 1),
            ("COVERAGE_MODE", "adaptve"),
            ("MAX_RESIDUAL_NORM", 0.0),
            ("MAX_RESIDUAL_NORM", float("inf")),
            ("RIDGE", -0.1),
            ("BANDWIDTH", float("nan")),
            ("RIDGE", True),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                options = model_options()
                setattr(options, key, value)
                state = torch.get_rng_state().clone()
                with self.assertRaises((TypeError, ValueError)):
                    ResidualTransportCLIP(copy.deepcopy(self.backbone), options=options)
                self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_adapter_rejects_boolean_and_string_geometry_numbers(self):
        for key in ("bandwidth", "ridge", "radius"):
            for value in (True, "0.25"):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(TypeError):
                        build_model(
                            copy.deepcopy(self.backbone), {key: value}, ["forest", "bridge"]
                        )

    def test_templates_cannot_drop_labels_or_add_formatting_fields(self):
        for template in ("a satellite image", "{} {}", "{} {other}", "{0}", "{!r}", "{0.upper} {}"):
            with self.subTest(template=template):
                with self.assertRaises(ValueError):
                    SharedPromptCLIP(copy.deepcopy(self.backbone), native_template=template)
                with self.assertRaises(ValueError):
                    build_model(
                        copy.deepcopy(self.backbone),
                        {"templates": [template]},
                        ["forest", "bridge"],
                    )
        for templates in ([], "a photo of a {}."):
            with self.assertRaises(ValueError):
                build_model(
                    copy.deepcopy(self.backbone), {"templates": templates}, ["forest", "bridge"]
                )

    def test_older_option_schema_defaults_do_not_mutate_callers(self):
        options = model_options()
        optional = (
            "RESIDUAL_BOUND",
            "COVERAGE_MODE",
            "AUXILIARY_SELF_DONOR",
            "TANGENT_PROJECTION",
            "AUXILIARY_WRONG_DONOR",
        )
        for key in optional:
            del options[key]
        before = options.clone()
        normalized = validated_model_options(options)
        self.assertEqual(options, before)
        self.assertEqual(normalized, model_options())


if __name__ == "__main__":
    unittest.main()
