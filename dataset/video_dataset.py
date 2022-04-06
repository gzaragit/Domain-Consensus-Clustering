import random
from os import listdir
from os.path import join
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .utils import GaussianBlur, RandAugment, get_mean_std, natural_keys
from random import choice


class VideoDataset(Dataset):
    def __init__(
        self,
        file_list,
        n_frames=16,
        n_clips=1,
        frame_size=224,
        transform=None,
        reorder_shape=True,
        epic_kitchens=False,
        remove_target_private=False,
        n_classes=12,
    ):
        super().__init__()

        self.file_list = file_list

        self.transform = transform

        self.n_frames = n_frames
        self.n_clips = n_clips
        self.reorder_shape = reorder_shape
        self.epic_kitchens = epic_kitchens
        self.mean, self.std = get_mean_std(file_list)

        if isinstance(frame_size, int):
            self.frame_size = (frame_size, frame_size)
        else:
            self.frame_size = frame_size

        self.videos_with_class = []
        if self.epic_kitchens:
            self.ek_videos = {}

        with open(file_list, "r") as filelist:
            for line in filelist:
                split_line = line.split()
                path = split_line[0]

                if self.epic_kitchens:
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    if remove_target_private:
                        if label < n_classes:
                            self.videos_with_class.append(
                                (path, start_frame, stop_frame, label)
                            )
                            kitchen = path.split("/")[-1]
                            if kitchen not in self.ek_videos:
                                kitchen_videos = self.find_frames(path)
                                kitchen_videos.sort(key=natural_keys)
                                self.ek_videos[kitchen] = kitchen_videos
                    else:
                        self.videos_with_class.append(
                            (path, start_frame, stop_frame, label)
                        )
                        kitchen = path.split("/")[-1]
                        if kitchen not in self.ek_videos:
                            kitchen_videos = self.find_frames(path)
                            kitchen_videos.sort(key=natural_keys)
                            self.ek_videos[kitchen] = kitchen_videos
                else:
                    label = int(split_line[1])
                    if remove_target_private:
                        if label < n_classes:
                            self.videos_with_class.append((path, label))
                    else:
                        self.videos_with_class.append((path, label))

        if "temporal" in self.augmentations:
            self.sampling_function = self.get_random_indices
        else:
            self.sampling_function = self.get_indices_clips

    def __len__(self):
        return len(self.videos_with_class)

    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def find_frames(self, video):
        """Finds frames from input sequence."""
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    def maybe_fix_gray(self, tensor: torch.Tensor) -> torch.Tensor:

        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    def spatial_jitter(self, frame, args, epsilon_value=5):
        epsilons = [epsilon_value, 0, -epsilon_value]
        return TF.resized_crop(
            frame,
            args["i"] + random.choice(epsilons),
            args["j"] + random.choice(epsilons),
            args["h"],
            args["w"],
            self.frame_size,
        )

    def channel_splitting(self, tensor_frame, args):
        replicate_channel = transforms.Lambda(
            lambda x: x[args["channel_index"], :, :].repeat(3, 1, 1)
        )
        augmented_tensor_frame = replicate_channel(tensor_frame)
        return augmented_tensor_frame

    def apply_transforms(self, frame, transform_args=None):



        return frame

    def convert_to_video(
        self, frames: List[Image.Image]
    ) -> torch.Tensor:
        """Generates a single transformation pipeline, applies it to all frames
        and returns a (n_clips, n_frames, c, h, w) tensor that represents the whole video.

        Args:
            List[Image.Image]: list of frames.
            List[Callable]: list of transformations

        Returns:
            torch.Tensor: 5d tensor that represents a video.
        """

        tensors = []
        for frame in frames:
            frame = self.transform(frame)
            tensors.append(frame)

        tensors = torch.stack(tensors)
        tensors = tensors.reshape(self.n_clips, self.n_frames, *tensors.size()[1:])

        if self.reorder_shape:
            tensors = tensors.permute(0, 2, 1, 3, 4)
        return tensors

    def load_frame(self, path, mode="RGB"):
        frame = Image.open(path).convert(mode)
        return frame

    def get_random_indices(self, num_frames: int) -> np.array:
        """Generates randomly distributed indices from the total number of frames.

        Args:
            num_frames (int): total number of frames.

        Returns:
            np.array: vector of all frame indexes.
        """

        indexes = np.sort(
            np.random.choice(num_frames, self.n_frames * self.n_clips, replace=True)
        )
        return indexes

    def get_indices(self, num_frames: int) -> np.array:
        """Generates uniformly distributed indices from the total number of frames.

        Args:
            num_frames (int): total number of frames.

        Returns:
            np.array: vector of all frame indexes.
        """

        tick = num_frames / self.n_frames
        indexes = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.n_frames)]
        )  # pick the central frame in each segment
        return indexes

    def get_indices_clips(self, num_frames):
        """Generates uniformly distributed indices from the total number of frames
        considering multiple clips.

        Args:
            num_frames (int): total number of frames.

        Returns:
            np.array: vector of all frame indexes.
        """

        num_frames_clip = num_frames // self.n_clips
        indexes = self.get_indices(num_frames_clip)
        indexes = np.tile(indexes, self.n_clips)
        for i in range(self.n_clips):
            indexes[i * self.n_frames : (i + 1) * self.n_frames] += num_frames_clip * i
        return indexes

    def get_transforms_args(self, frame_size):
        if self.frame_size[0] == 224:
            s = 256
        elif self.frame_size[0] == 112:
            s = 128
        else:
            raise Exception("Size is not supported")
        transform_args = {}
        if "spatial_jitter" in self.augmentations:
            dummy = TF.resize(Image.new("RGB", frame_size), s)
            i, j, h, w = transforms.RandomCrop.get_params(dummy, self.frame_size)
            transform_args["spatial_jitter"] = {"i": i, "j": j, "h": h, "w": w}
        if "channel_splitting" in self.augmentations:
            transform_args["channel_splitting"] = {
                "channel_index": random.choice(range(3))
            }
        return transform_args

    def __getitem__(self, index):

        if self.epic_kitchens:
            video, start_frame, stop_frame, y = self.videos_with_class[index]
        else:
            video, y = self.videos_with_class[index]

        # find frames
        if self.epic_kitchens:
            kitchen = video.split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[start_frame:stop_frame]
        else:
            frame_paths = self.find_frames(video)
            frame_paths.sort(key=natural_keys)

        indexes = self.sampling_function(len(frame_paths) - 1)
        frames = []
        for i in indexes:
            frames.append(self.load_frame(frame_paths[i]))

        video = self.convert_to_video(frames)

        return video, y
