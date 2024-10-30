from typing import Optional

from mindspore import Tensor, nn, ops

from ..schedulers import RFlowLossWrapper
from ..utils.model_utils import no_grad

__all__ = ["DiffusionWithLoss"]


class DiffusionWithLoss(nn.Cell):
    def __init__(
        self,
        network: RFlowLossWrapper,
        vae: Optional[nn.Cell] = None,
        text_encoder: Optional[nn.Cell] = None,
        scale_factor: float = 0.13025,
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
    ):
        super().__init__()

        if not text_emb_cached and text_encoder is None:
            raise ValueError("`text_encoder` must be provided when `text_emb_cached=False`.")
        if not video_emb_cached and vae is None:
            raise ValueError("`vae` must be provided when `video_emb_cached=False`.")

        self.network = network
        self.vae = vae
        self.text_encoder = text_encoder
        self.scale_factor = scale_factor
        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.vae is not None:
            for param in self.vae.trainable_params():
                param.requires_grad = False

        if self.text_encoder is not None:
            for param in self.text_encoder.trainable_params():
                param.requires_grad = False

    def get_condition_embeddings(self, text_tokens: Tensor) -> Tensor:
        if self.text_emb_cached:
            return text_tokens
        with no_grad():
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens))
        return text_emb

    def get_latents(self, video_tokens: Tensor) -> Tensor:
        if self.video_emb_cached:
            return video_tokens
        with no_grad():
            b, f, *_ = video_tokens.shape  # FIXME: no VideoAutoencoderKL in mindone.differs
            video_tokens = video_tokens.reshape(-1, *video_tokens.shape[2:])
            video_emb = ops.stop_gradient(self.vae.encode(video_tokens.astype(self.vae.dtype))[0] * self.scale_factor)
            video_emb = video_emb.reshape(b, f, *video_emb.shape[1:])
        return video_emb

    def construct(self, video_tokens: Tensor, ul2_tokens: Tensor, byt5_tokens: Tensor) -> Tensor:
        latent_embedding = self.get_latents(video_tokens)
        ul2_emb = self.get_condition_embeddings(ul2_tokens)
        byt5_emb = self.get_condition_embeddings(byt5_tokens)
        # FIXME: add metaclip
        metaclip_emb = ops.ones((byt5_emb.shape[0], 300, 1280), dtype=byt5_emb.dtype)
        return self.network(latent_embedding, ul2_emb, metaclip_emb, byt5_emb)
