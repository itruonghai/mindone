import os
import inspect
import gc
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, nn
from huggingface_hub import snapshot_download
from mindone.diffusers import AutoencoderKL
# from safetensors.mindspore import load_file
from mindnlp.transformers import Phi3Config

from OmniGen import OmniGen, OmniGenProcessor, OmniGenScheduler

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = OmniGenPipeline.from_pretrained(base_model)
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""

class OmniGenPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor
        self.model.set_train(False)
        self.vae.set_train(False)

    @classmethod
    def from_pretrained(cls, model_name, vae_path: str = None):
        config = Phi3Config.from_pretrained(model_name)
        model = OmniGen(config)
        # load_ckpt_params(model, '/mnt/disk2/nthai/mindone/examples/OmniGen/models/omnigen.ckpt')

        processor = OmniGenProcessor.from_pretrained(model_name)

        # Load VAE
        if os.path.exists(os.path.join(model_name, "vae")):
            vae = AutoencoderKL.from_pretrained(os.path.join(model_name, "vae"))
        elif vae_path is not None:
            vae = AutoencoderKL.from_pretrained(vae_path)
        else:
            print(f"No VAE found in {model_name}, downloading stabilityai/sdxl-vae")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

        return cls(vae, model, processor)

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.encode(x).latent_dist.sample()
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        x = x.astype(dtype)
        return x


    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        use_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: ms.dtype = ms.float32,
        seed: int = None,
        output_type: str = "pil",
    ):
        # Input validation
        if use_input_image_size_as_output:
            assert isinstance(prompt, str) and len(input_images) == 1, \
                "For matching output size to input, provide single image only"
        else:
            assert height % 16 == 0 and width % 16 == 0, \
                "Height and width must be multiples of 16"

        # Handle inputs
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None

        # Process inputs
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(
                self.processor.text_tokenizer,
                max_image_size=max_input_image_size
            )

        input_data = self.processor(
            prompt, input_images,
            height=height, width=width,
            use_img_cfg=use_img_guidance,
            separate_cfg_input=separate_cfg_infer,
            use_input_image_size_as_output=use_input_image_size_as_output
        )
        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1

        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data['input_pixel_values'][0][0].shape[-2:]
            else:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
        latent_size_h, latent_size_w = height // 8, width // 8

        # Initialize random latents
        if seed is not None:
            np.random.seed(seed)
        latents = np.random.randn(num_prompt, 4, latent_size_h, latent_size_w)
        latents = np.concatenate([latents] * (1 + num_cfg))
        latents = Tensor(latents, dtype=dtype)

        # Process input images
        input_img_latents = []
        if separate_cfg_infer:
            for temp_pixel_values in input_data['input_pixel_values']:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(Tensor(img), dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data['input_pixel_values']:
                img = self.vae_encode(Tensor(img), dtype)
                input_img_latents.append(img)

        # Prepare model inputs
        model_kwargs = dict(
            input_ids=input_data['input_ids'],
            input_img_latents=input_img_latents,
            input_image_sizes=input_data['input_image_sizes'],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data["position_ids"],
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
        )

        # Choose generation function
        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        # Generate image
        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache)
        samples = samples.split(axis=0, output_num=1+num_cfg)[0]

        # Decode latents
        samples = samples.astype(ms.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor

        samples = self.vae.decode(samples).sample
        samples = (samples * 0.5 + 0.5).clip(0, 1)

        # Convert to output format
        if output_type == "pt":
            output_images = samples
        else:
            samples = (samples * 255).astype(ms.uint8)
            samples = samples.transpose(0, 2, 3, 1)
            samples = samples.asnumpy()
            output_images = []
            for sample in samples:
                output_images.append(Image.fromarray(sample))

        return output_images
    

def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> nn.Cell:
    if isinstance(ckpt, str):
        print(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if not (len(param_not_load) == len(ckpt_not_load) == 0):
        print(
            "Exist ckpt params not loaded: {} (total: {}), or net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        )
    return model
