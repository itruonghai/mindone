from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, List
import gc

import mindspore as ms
import mindspore.ops as ops
from mindnlp.transformers.cache_utils import Cache, DynamicCache

class OmniGenCache(DynamicCache):
    """Custom cache implementation for OmniGen"""
    def __init__(self, num_tokens_for_img: int) -> None:
        super().__init__()
        self.num_tokens_for_img = num_tokens_for_img

    def update(
        self,
        key_states: ms.Tensor, 
        value_states: ms.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Updates cache with new key/value states"""
        if len(self.key_cache) < layer_idx:
            raise ValueError("Cache does not support skipping layers. Use DynamicCache.")
            
        elif len(self.key_cache) == layer_idx:
            # Only cache condition tokens
            key_states = key_states[..., :-(self.num_tokens_for_img+1), :]
            value_states = value_states[..., :-(self.num_tokens_for_img+1), :]

            # Update seen tokens count
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
                
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
            
        else:
            # Only cache condition tokens
            key_tensor, value_tensor = self[layer_idx]
            k = ops.concat([key_tensor, key_states], axis=-2)
            v = ops.concat([value_tensor, value_states], axis=-2)
            return k, v

class OmniGenScheduler:
    """Scheduler for the diffusion process"""
    def __init__(self, num_steps: int=50, time_shifting_factor: int=1):
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor

        t = ms.numpy.linspace(0, 1, num_steps+1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = t

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        """Crop position IDs for cache"""
        if isinstance(position_ids, list):
            for i in range(len(position_ids)):
                position_ids[i] = position_ids[i][:, -(num_tokens_for_img+1):]
        else:
            position_ids = position_ids[:, -(num_tokens_for_img+1):]
        return position_ids

    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        """Crop attention mask for cache"""
        if isinstance(attention_mask, list):
            return [x[..., -(num_tokens_for_img+1):, :] for x in attention_mask]
        return attention_mask[..., -(num_tokens_for_img+1):, :]

    def crop_cache(self, cache, num_tokens_for_img):
        """Crop cache to remove unneeded tokens"""
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :-(num_tokens_for_img+1), :]
            cache.value_cache[i] = cache.value_cache[i][..., :-(num_tokens_for_img+1), :]
        return cache

    def __call__(self, z, func, model_kwargs, use_kv_cache: bool=True):
        """Run the diffusion process"""
        num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4
        
        # Initialize cache
        # if isinstance(model_kwargs['input_ids'], list):
        #     cache = [OmniGenCache(num_tokens_for_img) for _ in range(len(model_kwargs['input_ids']))] if use_kv_cache else None
        # else:
        #     cache = OmniGenCache(num_tokens_for_img) if use_kv_cache else None
        
        cache = None
        # Run diffusion steps
        for i in tqdm(range(self.num_steps)):
            timesteps = ops.zeros(len(z), dtype=z.dtype) + self.sigma[i]

            #TODO delete _z
            pred, cache = func(z, timesteps, past_key_values=cache, **model_kwargs)
            
            sigma_next = self.sigma[i+1]
            sigma = self.sigma[i]
            z = z + (sigma_next - sigma) * pred

            # Update model kwargs for caching after first step
            if i == 0 and use_kv_cache:
                num_tokens_for_img = z.shape[-1] * z.shape[-2] // 4
                if isinstance(cache, list):
                    model_kwargs['input_ids'] = [None] * len(cache)
                else:
                    model_kwargs['input_ids'] = None

                model_kwargs['position_ids'] = self.crop_position_ids_for_cache(
                    model_kwargs['position_ids'], num_tokens_for_img)
                model_kwargs['attention_mask'] = self.crop_attention_mask_for_cache(
                    model_kwargs['attention_mask'], num_tokens_for_img)

        del cache
        return z