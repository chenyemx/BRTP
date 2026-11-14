"""Shared context for native class names; only ctx receives gradients.

The transformer remains differentiable with respect to the shared context.
Notation: n_ctx is M, ctx is Theta (row m is theta_m), and initial_ctx
is Theta_0. M counts learnable tokens, not CLIP's full sequence length of 77.
"""

import torch
from torch import nn
from torch.nn import functional as F
from clip import clip
from brtp.config import validate_template


class PromptLearner(nn.Module):
    def __init__(self, backbone, n_ctx=4, ctx_init="a photo of a"):
        super().__init__()
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int):
            raise TypeError("N_CTX must be an integer")
        if not 1 <= n_ctx <= 32:
            raise ValueError("N_CTX must be between 1 and 32")
        if not isinstance(ctx_init, str):
            raise TypeError(
                "CTX_INIT must be a string; use an empty string for random initialization"
            )
        # Declared recipe metadata only; checkpoint keys remain ctx/initial_ctx.
        self.ctx_init = ctx_init
        width = backbone.ln_final.weight.numel()
        vectors = torch.empty(n_ctx, width, dtype=torch.float32)
        nn.init.normal_(vectors, std=0.02)
        if ctx_init:
            ids = clip.tokenize(ctx_init, truncate=False).to(backbone.token_embedding.weight.device)
            length = int(ids[0].argmax()) - 1
            if length < n_ctx:
                raise ValueError("CTX_INIT must tokenize to at least N_CTX tokens")
            with torch.no_grad():
                vectors.copy_(backbone.token_embedding(ids)[0, 1 : 1 + n_ctx])
        self.ctx = nn.Parameter(vectors)


class SharedPromptCLIP(nn.Module):
    def __init__(
        self,
        backbone,
        n_ctx=4,
        ctx_init="a photo of a",
        *,
        renderer_version="native_query_v1",
        native_template="{}.",
        teacher_template="a photo of a {}.",
    ):
        super().__init__()
        if renderer_version != "native_query_v1":
            raise ValueError("BRTP supports the native_query_v1 class-name renderer only")
        validate_template(native_template)
        validate_template(teacher_template)
        self.backbone = backbone.requires_grad_(False).eval()
        self.prompt_learner = PromptLearner(backbone, n_ctx, ctx_init)
        self.n_ctx = n_ctx
        self.renderer_version = renderer_version
        self.native_template = native_template
        self.teacher_template = teacher_template
        self._text_cache = {}
        self._frozen_cache = {}
        self.prompt_learner.register_load_state_dict_post_hook(self._prompt_loaded)

    def clear_derived_cache(self):
        """Discard tensors derived from the frozen backbone or prompt state."""
        self._text_cache.clear()
        self._frozen_cache.clear()

    def _prompt_loaded(self, _module, _incompatible_keys):
        self.clear_derived_cache()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.clear_derived_cache()
        return result

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def render_texts(self, texts, zero_shot=False, text_kind="native"):
        """Render native labels exactly as in the frozen training protocol."""
        if text_kind != "native":
            raise ValueError("BRTP text_kind must be native; relation queries are unsupported")
        if isinstance(texts, str) or not texts:
            raise ValueError("texts must be a nonempty sequence of strings")
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Every text must be a nonempty string")
        template = self.teacher_template if zero_shot else self.native_template
        rendered = [template.format(text.replace("_", " ")) for text in texts]
        if not zero_shot:
            rendered = [("X " * self.n_ctx) + text for text in rendered]
        return rendered

    def text_audit(self, texts, text_kind="native", zero_shot=False):
        """Return JSON-compatible rendering/token records for supplied texts.

        Token length includes SOT and EOT; EOT positions are zero-based. The
        caller supplies source texts during training. This function neither
        discovers candidate vocabularies nor accesses any dataset.
        """
        texts = tuple(texts) if not isinstance(texts, str) else texts
        rendered = self.render_texts(texts, zero_shot=zero_shot, text_kind=text_kind)
        ids = clip.tokenize(rendered, truncate=False)
        return [
            dict(
                input=text,
                rendered=rendered[index],
                text_kind=text_kind,
                zero_shot=bool(zero_shot),
                renderer_version=self.renderer_version,
                token_length=int(ids[index].argmax()) + 1,
                eot_position=int(ids[index].argmax()),
            )
            for index, text in enumerate(texts)
        ]

    def _text_inputs(self, texts, zero_shot=False, text_kind="native"):
        texts = tuple(texts) if not isinstance(texts, str) else texts
        # Validation also occurs for cache hits; the rendering definition is
        # shared with text_audit so saved token records match actual inputs.
        rendered = self.render_texts(texts, zero_shot=zero_shot, text_kind=text_kind)
        device = self.prompt_learner.ctx.device
        key = (
            tuple(texts),
            bool(zero_shot),
            text_kind,
            str(device),
            str(self.backbone.token_embedding.weight.dtype),
            self.renderer_version,
            self.native_template,
            self.teacher_template,
            self.n_ctx,
        )
        if key not in self._text_cache:
            ids = clip.tokenize(rendered, truncate=False).to(device)
            with torch.no_grad():
                embeddings = self.backbone.token_embedding(ids).detach()
            self._text_cache[key] = (ids, embeddings)
        return self._text_cache[key]

    def encode_texts(self, texts, zero_shot=False, text_kind="native"):
        ids, embeddings = self._text_inputs(texts, zero_shot, text_kind)
        if zero_shot:
            with torch.no_grad():
                return F.normalize(self.backbone.encode_text(ids).float(), dim=-1)
        return self._encode_context(ids, embeddings, self.prompt_learner.ctx)

    def encode_with_context(self, texts, context):
        """Encode a supplied context using the same arithmetic as learned prompts."""
        ids, embeddings = self._text_inputs(texts)
        return self._encode_context(ids, embeddings, context)

    def _encode_context(self, ids, embeddings, context):
        ctx = context.unsqueeze(0).expand(len(ids), -1, -1)
        x = torch.cat((embeddings[:, :1], ctx, embeddings[:, 1 + self.n_ctx :]), dim=1)
        x = x + self.backbone.positional_embedding.to(x.dtype)
        x = self.backbone.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.backbone.ln_final(x)
        x = x[torch.arange(len(ids), device=context.device), ids.argmax(dim=-1)]
        return F.normalize((x @ self.backbone.text_projection).float(), dim=-1)

    @torch.no_grad()
    def encode_image(self, images):
        return F.normalize(self.backbone.encode_image(images).float(), dim=-1)

    def classify_features(self, features, texts, zero_shot=False, text_kind="native"):
        text = self.encode_texts(texts, zero_shot=zero_shot, text_kind=text_kind)
        return self.backbone.logit_scale.exp().float() * features.float() @ text.float().t()

    def forward(self, images, texts, query_texts=None, zero_shot=False):
        if query_texts is not None:
            raise ValueError("BRTP accepts native class names only")
        features = self.encode_image(images)
        logits = self.classify_features(features, texts, zero_shot=zero_shot)
        return logits, None
