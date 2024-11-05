import csv
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader

from .transforms import ResizeCrop

_logger = logging.getLogger(__name__)


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


class ImageVideoDataset(BaseDataset):
    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Optional[Union[str, Dict[str, str]]] = None,
        vae_latent_folder: Optional[str] = None,
        vae_downsample_rate: float = 8.0,
        vae_scale_factor: float = 0.18215,
        target_size: Optional[Tuple[int, int]] = None,
        sample_n_frames: int = 17,
        sample_stride: int = 1,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        t_compress_func: Optional[Callable[[int], int]] = None,
        filter_data: bool = False,
        apply_transforms_dataset: bool = False,
        *,
        output_columns: List[str],
    ):
        if text_emb_folder is None:
            raise NotImplementedError(
                "Text embedding during training is not supported, please provide `text_emb_folder`."
            )

        self._data = self._read_data(video_folder, csv_path, text_emb_folder, vae_latent_folder, filter_data)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._text_emb_folder = text_emb_folder
        self._vae_latent_folder = vae_latent_folder
        self._vae_downsample_rate = vae_downsample_rate
        self._vae_scale_factor = vae_scale_factor
        self._fmask_gen = frames_mask_generator
        self._t_compress_func = t_compress_func or (lambda x: x)

        self.output_columns = output_columns

        self._transforms = (
            self.train_transforms(target_size, interpolation=cv2.INTER_AREA) if apply_transforms_dataset else None
        )

        # prepare replacement data in case the loading of a sample fails
        self._prev_ok_sample = self._get_replacement()
        self._require_update_prev = False

    @staticmethod
    def _read_data(
        data_dir: str,
        csv_path: str,
        text_emb_folder: Optional[Union[str, Dict[str, str]]] = None,
        vae_latent_folder: Optional[str] = None,
        filter_data: bool = False,
    ) -> List[dict]:
        def _filter_data(sample_):
            if not os.path.isfile(sample_["video"]):
                _logger.warning(f"Video not found: {sample_['video']}")
                return None
            if "text_emb" in sample_:
                if isinstance(sample_["text_emb"], str) and not os.path.isfile(sample_["text_emb"]):
                    _logger.warning(f"Text embedding not found: {sample_['text_emb']}")
                    return None
                else:
                    for name, path in sample_["text_emb"].items():
                        if not os.path.isfile(sample_["text_emb"][name]):
                            _logger.warning(f"Text embedding not found: {sample_['text_emb'][name]}")
                            return None
            if "vae_latent" in sample_ and not os.path.isfile(sample_["vae_latent"]):
                _logger.warning(f"Text embedding not found: {sample_['vae_latent']}")
                return None
            return sample_

        with open(csv_path, "r") as csv_file:
            try:
                data = []
                for item in csv.DictReader(csv_file):
                    sample = {**item, "video": os.path.join(data_dir, item["video"])}
                    if text_emb_folder:
                        if isinstance(text_emb_folder, str):
                            sample["text_emb"] = os.path.join(text_emb_folder, Path(item["video"]).with_suffix(".npz"))
                        else:
                            sample["text_emb"] = {
                                name: os.path.join(path, Path(item["video"]).with_suffix(".npz"))
                                for name, path in text_emb_folder.items()
                            }
                    if vae_latent_folder:
                        sample["vae_latent"] = os.path.join(vae_latent_folder, Path(item["video"]).with_suffix(".npz"))
                    data.append(sample)
            except KeyError as e:
                _logger.error(f"CSV file requires `video` (file paths) column, but got {list(item.keys())}")
                raise e

        if filter_data:
            with ThreadPoolExecutor(max_workers=10) as executor:
                data = [
                    item
                    for item in tqdm(executor.map(_filter_data, data), total=len(data), desc="Filtering data")
                    if item is not None
                ]

        _logger.info(f"Number of data samples: {len(data)}")
        return data

    def _get_replacement(self, max_attempts: int = 100) -> Tuple[np.ndarray, ...]:
        attempts, error = min(max_attempts, len(self)), None
        for idx in range(attempts):
            try:
                return self._get_item(idx)
            except Exception as e:
                error = e
                _logger.debug(f"Failed to load a replacement sample: {repr(e)}")

        raise RuntimeError(f"Fail to load a replacement sample in {attempts} attempts. Error: {repr(error)}")

    def _get_item(self, idx: int, thw: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, ...]:
        data = self._data[idx].copy()
        num_frames = self._frames

        if self._text_emb_folder:
            if isinstance(data["text_emb"], str):
                with np.load(data["text_emb"]) as td:
                    data.update({"caption": td["text_emb"], "mask": td["mask"]})
            else:
                for enc_name, path in data["text_emb"].items():
                    with np.load(path) as td:
                        data.update({enc_name + "_caption": td["text_emb"], enc_name + "_mask": td["mask"]})

        if self._vae_latent_folder:
            # TODO: add support for images
            vae_latent_data = np.load(data["vae_latent"])
            latent_mean, latent_std = vae_latent_data["latent_mean"], vae_latent_data["latent_std"]
            if len(latent_mean) < self._min_length:
                raise ValueError(f"Video is too short: {data['video']}")

            if "fps" not in data:
                if "fps" in vae_latent_data:
                    data["fps"] = vae_latent_data["fps"]
                else:
                    with VideoReader(data["video"]) as reader:
                        data["fps"] = reader.fps
            data["fps"] = np.array(data["fps"] / self._stride, dtype=np.float32)

            start_pos = random.randint(0, len(latent_mean) - self._min_length)
            batch_index = np.linspace(start_pos, start_pos + self._min_length - 1, num_frames, dtype=int)

            latent_mean, latent_std = latent_mean[batch_index], latent_std[batch_index]
            vae_latent = latent_mean + latent_std * np.random.standard_normal(latent_mean.shape)
            data["video"] = vae_latent * self._vae_scale_factor

        else:
            if data["video"].lower().endswith(IMAGE_EXT):
                num_frames = 1
                data["fps"] = np.array(120, dtype=np.float32)  # FIXME: extract as IMG_FPS
                data["video"] = cv2.cvtColor(cv2.imread(data["video"]), cv2.COLOR_BGR2RGB)
            else:
                with VideoReader(data["video"]) as reader:
                    min_length = self._min_length
                    if thw is not None:
                        num_frames, *data["size"] = thw
                        min_length = (num_frames - 1) * self._stride + 1
                    if len(reader) < min_length:
                        raise ValueError(f"Video is too short: {data['video']}")
                    start_pos = random.randint(0, len(reader) - min_length)
                    data["video"] = reader.fetch_frames(num=num_frames, start_pos=start_pos, step=self._stride)
                    data["fps"] = np.array(reader.fps / self._stride, dtype=np.float32)

        data["num_frames"] = np.array(num_frames, dtype=np.float32)

        if self._fmask_gen is not None:
            # return frames mask with respect to the VAE's latent temporal compression
            data["frames_mask"] = self._fmask_gen(self._t_compress_func(num_frames))

        if self._transforms:
            data = self._apply_transforms(data)

        return tuple(data[c] for c in self.output_columns)

    def get_bucket(self, thw: Tuple[int, int, int], sample_ids: List[int]) -> Tuple[np.ndarray, ...]:
        batch = [self._get_item(sample_id, thw) for sample_id in sample_ids]
        return tuple(np.stack(item) for item in map(list, zip(*batch)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        try:
            sample = self._get_item(idx)
            if self._require_update_prev:
                self._prev_ok_sample = sample
                self._require_update_prev = False
        except Exception as e:
            _logger.warning(f"Failed to fetch sample #{idx}, the video will be replaced. Error: {e}")
            sample = self._prev_ok_sample
            self._require_update_prev = True

        return sample

    def __len__(self):
        return len(self._data)

    def _apply_transforms(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for transform in self._transforms:
            input_data = tuple(data[column] for column in transform["input_columns"])
            for op in transform["operations"]:
                input_data = op(*input_data)
                if not isinstance(input_data, tuple):  # wrap numpy array in a tuple
                    input_data = (input_data,)
            data.update(zip(transform.get("output_columns", transform["input_columns"]), input_data))
        return data

    def train_transforms(
        self,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        tokenizer: Optional[Callable[[str], np.ndarray]] = None,
    ) -> List[dict]:
        transforms = []
        vae_downsample_rate = self._vae_downsample_rate

        if not self._vae_latent_folder:
            vae_downsample_rate = 1
            transforms.append(
                {
                    "operations": [
                        ResizeCrop(target_size, interpolation=interpolation),
                        lambda x: x.astype(np.float32) / 127.5 - 1,
                        lambda x: np.transpose(x, (0, 3, 1, 2)),
                    ],
                    "input_columns": ["video"],
                }
            )
        # the followings are not transformation for video frames, can be excluded
        transforms.append(
            {
                "operations": [
                    lambda video: (
                        video,  # need to return the video itself to preserve the column
                        np.array(video.shape[-2] * vae_downsample_rate, dtype=np.float32),
                        np.array(video.shape[-1] * vae_downsample_rate, dtype=np.float32),
                        np.array(video.shape[-2] / video.shape[-1], dtype=np.float32),
                    )
                ],
                "input_columns": ["video"],
                "output_columns": ["video", "height", "width", "ar"],
            }
        )

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
