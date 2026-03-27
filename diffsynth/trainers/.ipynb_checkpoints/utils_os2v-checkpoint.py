import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import functional as F
from accelerate.utils import DistributedDataParallelKwargs
import ijson
from typing import Dict, Any, Iterator, List, Tuple
import decord
from decord import VideoReader
import tempfile, os, requests
import random
import numpy as np
from torchvision import transforms
import gc
from pycocotools import mask as mask_util
from torch.utils.data._utils.collate import default_collate
from accelerate.state import PartialState
from accelerate.utils import broadcast_object_list
from accelerate.utils import DataLoaderConfiguration
from torch.utils.data import get_worker_info

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    # def crop_and_resize(self, image, target_height, target_width):
    #     width, height = image.size
    #     scale = max(target_width / width, target_height / height)
    #     image = torchvision.transforms.functional.resize(
    #         image,
    #         (round(height*scale), round(width*scale)),
    #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    #     )
    #     image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    #     return image
    def crop_and_resize(self, image, target_width, target_height, pad_color=(255, 255, 255)):
        """
        Args:
            image: PIL.Image
            target_height, target_width: 目标高宽
            pad_color: padding 颜色，默认为白色
        Returns:
            PIL.Image，保持比例缩放并以 pad 填充到指定大小
        """
        w, h = image.size

        # 统一用缩放系数（短边对齐，保证不超出目标框）
        scale = min(target_width / w, target_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # 等比缩放（PIL 直接支持，antialias 更好看）
        resized = F.resize(
            image,
            [new_h, new_w],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        # 计算四边 padding（左、上、右、下），确保总和正好补齐到目标尺寸
        pad_left   = (target_width  - new_w) // 2
        pad_top    = (target_height - new_h) // 2
        pad_right  = target_width  - new_w - pad_left
        pad_bottom = target_height - new_h - pad_top

        # 直接对 PIL.Image 进行 pad
        out = F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=pad_color)

        return out
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



def open_with_decord_from_url(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk: tmp.write(chunk)
        tmp.flush(); tmp.close()
        return tmp.name   # 这个路径直接给 decord.VideoReader
    except:
        try: os.unlink(tmp.name)
        except: pass
        raise RuntimeError(f"failed to download {url}")

def rle_to_mask(rle, img_width, img_height):
    rle_obj = {"counts": rle["counts"].encode("utf-8"), "size": [img_height, img_width]}
    return mask_util.decode(rle_obj)

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        json_mode: str = "array",               # 'array' 或 'kv'
        sample_stride=3,
        args=None,
    ):
        if args is not None:
            # base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path #json path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            json_mode = args.json_mode
        
        # self.base_path = base_path
        self.num_frames = num_frames
        if metadata_path is None:
            self.metadata_paths = None
        elif isinstance(metadata_path, (list, tuple)):
            self.metadata_paths = list(metadata_path)
        else:
            # 字符串：支持逗号分隔
            self.metadata_paths = [
                p.strip() for p in str(metadata_path).split(",") if p.strip()
            ]
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.json_mode = json_mode
        self.sample_stride = sample_stride
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if self.metadata_paths is None:
            raise ValueError("metadata_path is required for streaming JSON.")    
    
        
        
    # def crop_and_resize(self, image, target_height, target_width):
    #     width, height = image.size
    #     scale = max(target_width / width, target_height / height)
    #     image = torchvision.transforms.functional.resize(
    #         image,
    #         (round(height*scale), round(width*scale)),
    #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    #     )
    #     image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    #     return image
    def crop_and_resize(self, image, target_width, target_height, pad_color=(255, 255, 255)):
        """
        Args:
            image: PIL.Image
            target_height, target_width: 目标高宽
            pad_color: padding 颜色，默认为白色
        Returns:
            PIL.Image，保持比例缩放并以 pad 填充到指定大小
        """
        w, h = image.size

        # 统一用缩放系数（短边对齐，保证不超出目标框）
        scale = min(target_width / w, target_height / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # 等比缩放（PIL 直接支持，antialias 更好看）
        resized = F.resize(
            image,
            [new_h, new_w],
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        # 计算四边 padding（左、上、右、下），确保总和正好补齐到目标尺寸
        pad_left   = (target_width  - new_w) // 2
        pad_top    = (target_height - new_h) // 2
        pad_right  = target_width  - new_w - pad_left
        pad_bottom = target_height - new_h - pad_top

        # 直接对 PIL.Image 进行 pad
        out = F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=pad_color)

        return out
        
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames

    def load_image_dir(self, dir_path):
        import glob, re, os
        # 支持 jpg/png/... 扩展名
        pats = [os.path.join(dir_path, f"*.{ext}") for ext in self.image_file_extension]
        files = []
        for p in pats:
            files.extend(glob.glob(p))
        # 自然排序
        def _natural_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(s))]
        files = sorted(set(files), key=_natural_key)

        frames = []
        for fp in files:
            img = Image.open(fp).convert("RGB")
            img = self.crop_and_resize(img, *self.get_height_width(img))
            frames.append(img)
            # img.close()
        return frames

    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if os.path.isdir(file_path):
            return self.load_image_dir(file_path)  # 路径是目录 -> 加载所有图片
        elif self.is_image(file_path):
            return self.load_image(file_path)      # 单文件 -> 加载一张
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    # ---------- ijson 流式解析 ----------
    def _iter_json_array(self) -> Iterator[Dict[str, Any]]:
        for path in self.metadata_paths:
            with open(path, "r", encoding="utf-8") as f:
                for rec in ijson.items(f, "item"):
                    yield rec

    def _iter_json_kv(self) -> Iterator[Dict[str, Any]]:
        for path in self.metadata_paths:
            with open(path, "r", encoding="utf-8") as f:
                for key, rec in ijson.kvitems(f, ""):
                    if isinstance(rec, dict):
                        rec.setdefault("key", key)
                    yield rec

    def _iter_json_multiv(self) -> Iterator[Dict[str, Any]]:
        for path in self.metadata_paths:
            with open(path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "", multiple_values=True, use_float=True):
                    if isinstance(obj, dict):
                        yield obj


    def _iter_records(self) -> Iterator[Dict[str, Any]]:
        if self.json_mode == "array":
            yield from self._iter_json_array()
        elif self.json_mode == "kv":
            yield from self._iter_json_kv()
        elif self.json_mode == "multiv":
            yield from self._iter_json_multiv()
        else:
            raise ValueError("json_mode 必须为 'array' 或 'kv'（不使用 jsonl）。")
    
    def _get_frame_indices_adjusted(self, video_length, n_frames):
        indices = list(range(video_length))
        additional_frames_needed = n_frames - video_length

        repeat_indices = []
        for i in range(additional_frames_needed):
            index_to_repeat = i % video_length
            repeat_indices.append(indices[index_to_repeat])

        all_indices = indices + repeat_indices
        all_indices.sort()

        return all_indices

    def _generate_frame_indices(self, valid_start, valid_end, n_frames, sample_stride):
        adjusted_length = valid_end - valid_start

        if adjusted_length <= n_frames:
            frame_indices = self._get_frame_indices_adjusted(adjusted_length, n_frames)
            frame_indices = [i + valid_start for i in frame_indices]
        else:
            clip_length = min(adjusted_length, (n_frames - 1) * sample_stride + 1)
            start_idx = random.randint(valid_start, valid_end - clip_length)
            frame_indices = np.linspace(
                start_idx, start_idx + clip_length - 1, n_frames, dtype=int
            ).tolist()

        return frame_indices
    
    def _short_resize_and_crop(self, frames, target_width, target_height):
        T, C, H, W = frames.shape
        aspect_ratio = W / H

        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            if new_height < target_height:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            if new_width < target_width:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)

        resize_transform = transforms.Resize((new_height, new_width))
        crop_transform = transforms.CenterCrop((target_height, target_width))

        frames_tensor = frames  # (T, C, H, W)
        resized_frames = resize_transform(frames_tensor)
        cropped_frames = crop_transform(resized_frames)
        pil_frames = []
        for i in range(cropped_frames.shape[0]):
            frame = cropped_frames[i]  # (C, H, W)
            frame = F.to_pil_image(frame)  # Tensor -> PIL
            pil_frames.append(frame)

        return pil_frames

    # def _short_resize_and_crop(self, frames, target_width, target_height):
    #     """
    #     Args:
    #         frames: Tensor, shape (T, C, H, W)
    #         target_width, target_height: 目标尺寸
    #     Returns:
    #         List[PIL.Image]  每帧都按短边等比缩放并用白色pad到指定大小
    #     """
    #     import torchvision
    #     from torchvision.transforms import functional as F
    #     from torchvision.transforms import InterpolationMode

    #     T, C, H, W = frames.shape
    #     pad_color = (255, 255, 255)  # 与上面函数保持一致，白色背景

    #     pil_frames = []
    #     for i in range(T):
    #         # (C, H, W) -> PIL.Image (RGB/单通道由输入决定)
    #         frame_tensor = frames[i]
    #         img = F.to_pil_image(frame_tensor)

    #         w, h = img.size
    #         # 与上面 crop_and_resize 相同：短边对齐，确保不超过目标框
    #         scale = min(target_width / w, target_height / h)
    #         new_w = max(1, int(round(w * scale)))
    #         new_h = max(1, int(round(h * scale)))

    #         # 等比缩放 + 抗锯齿
    #         resized = F.resize(
    #             img,
    #             [new_h, new_w],
    #             interpolation=InterpolationMode.BILINEAR,
    #             antialias=True
    #         )

    #         # 计算四边 padding（左、上、右、下），总计补齐到目标尺寸
    #         pad_left   = (target_width  - new_w) // 2
    #         pad_top    = (target_height - new_h) // 2
    #         pad_right  = target_width  - new_w - pad_left
    #         pad_bottom = target_height - new_h - pad_top

    #         # 居中填充到目标大小（白色）
    #         out = F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=pad_color)
    #         pil_frames.append(out)

    #     return pil_frames

    
    def get_cropped_subject_image(
        self,
        input_image,
        annotation_idx,
        class_names,
        mask_data,
        image_width,
        image_height,
        bbox_data,
        use_bbox=False,
        as_float: bool = True,
    ):
        class_name = class_names[f"{annotation_idx}"]["class_name"]
        record = bbox_data[int(annotation_idx) - 1]
        gme_score = record.get("gme_score", 0.0)
        aes_score = record.get("aes_score", 0.0)
        if gme_score == 0.0:
            print("bad data")
            # assert 2==1
        # gme_score = bbox_data[int(annotation_idx) - 1]["gme_score"]
        # aes_score = bbox_data[int(annotation_idx) - 1]["aes_score"]

        if use_bbox:
            bbox = bbox_data[int(annotation_idx) - 1]["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]

            x_min = int(max(x_min, 0))
            y_min = int(max(y_min, 0))
            x_max = int(min(x_max, image_width - 1))
            y_max = int(min(y_max, image_height - 1))

            resized_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
        else:
            mask_rle = mask_data[annotation_idx]
            mask = rle_to_mask(mask_rle, image_width, image_height)

            # Find the bounding box of the mask
            rows, cols = np.where(mask == 1)
            if len(rows) == 0 or len(cols) == 0:
                bbox = bbox_data[int(annotation_idx) - 1]["bbox"]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[2]
                y_max = bbox[3]

                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(x_max, image_width - 1))
                y_max = int(min(y_max, image_height - 1))

                resized_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
            else:
                y_min, y_max = np.min(rows), np.max(rows)
                x_min, x_max = np.min(cols), np.max(cols)

                # Adjust if the region goes out of bounds
                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(x_max, image_width - 1))
                y_max = int(min(y_max, image_height - 1))

                # Crop the region from the original image and mask
                cropped_image = input_image[y_min : y_max + 1, x_min : x_max + 1]
                cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

                # Create a white background of the same size as the crop
                white_background = np.ones_like(cropped_image) * 255

                # Apply the mask to the cropped image
                white_background[cropped_mask == 1] = cropped_image[cropped_mask == 1]
                resized_image = white_background

        # pil_image = Image.fromarray(
        #     cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        # ).convert("RGB")
        pil_image = Image.fromarray(resized_image.astype(np.uint8), mode="RGB").convert("RGB")
        # pil_image.save(f"{class_name}_{i}_ratio{crop_ratio:.4f}.png")  # For sanity check

        crop_ratio = (pil_image.size[0] * pil_image.size[1]) / (
            image_height * image_width
        )

        return pil_image, class_name, gme_score, aes_score, crop_ratio


    # def short_resize_and_crop_pil(self, image: Image.Image, target_width: int, target_height: int):
    #     W, H = image.size
    #     aspect_ratio = W / H

    #     if aspect_ratio > target_width / target_height:
    #         new_width = target_width
    #         new_height = int(target_width / aspect_ratio)
    #         if new_height < target_height:
    #             new_height = target_height
    #             new_width = int(target_height * aspect_ratio)
    #     else:
    #         new_height = target_height
    #         new_width = int(target_height * aspect_ratio)
    #         if new_width < target_width:
    #             new_width = target_width
    #             new_height = int(target_width / aspect_ratio)

    #     resize_transform = transforms.Resize((new_height, new_width))
    #     crop_transform = transforms.CenterCrop((target_height, target_width))

    #     resized_image = resize_transform(image)
    #     cropped_image = crop_transform(resized_image)

    #     return cropped_image

    
    def _valid_sample(self, s2v_data: Dict[str, Any]) -> bool:
        # video: List[PIL.Image] 或 Tensor/List[Tensor] 都可 —— 只要非空
        if "video" not in s2v_data or s2v_data["video"] is None:
            return False
        v = s2v_data["video"]
        if (hasattr(v, "__len__") and len(v) == 0):
            return False

        # subject_image: 至少有一张（根据你训练的需要调整）
        if "subject_image" not in s2v_data or s2v_data["subject_image"] is None:
            return False
        if len(s2v_data["subject_image"]) == 0:
            return False

        # prompt: 非空字符串（如果不是必须，可以放宽）
        if "prompt" not in s2v_data or not isinstance(s2v_data["prompt"], str) or len(s2v_data["prompt"].strip()) == 0:
            return False

        return True

    # ---------- IterableDataset 接口 ----------
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # state = PartialState()                 # 加速器的全局进程状态
        # rank = state.process_index
        # world = state.num_processes
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        for _ in range(self.repeat):
            for idx, rec in enumerate(self._iter_records()):
                # print(rec)
                if (idx % num_workers) != worker_id:
                    continue
                s2v_data = {}
                
                data = rec.copy()
                # print("_________________________________",data["video_id"])
                # if data["video_id"] =="Rd4O2umVFkE_segment_3_step1-0-75_step2-0-75_step4_step5_step6":
                    # print("fixed the bug by get")
                #     save_path = "/root/paddlejob/workspace/env_run/wanglei/debug_outputs/pro_data.json"
                #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #     with open(save_path, "w", encoding="utf-8") as f:
                #         json.dump(data, f, ensure_ascii=False, indent=2)
                #     print(f"[SAVE] Problematic data saved to: {save_path}")
                # else:
                #     # print(data)
                #     # 跳过其他样本
                    # continue
                # print(data)
                ok = True
                # print(self.data_file_keys)
                # for key in self.data_file_keys:
                #video
                decord.bridge.set_bridge("torch")
                try:
                    local_path = open_with_decord_from_url(data["bos_url"])
                    vr = VideoReader(local_path, num_threads=4)
                except Exception as e:
                    print(f"[WARN] skip sample (video_id={data.get('video_id')}) due to: {e}")
                    continue

                # crop (remove watermark) & cut (remove transition)
                s_x, e_x, s_y, e_y = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["metadata"]["crop"]
                start_frame = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["metadata"]["face_cut"][0]
                end_frame = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["metadata"]["face_cut"][1]
                frame_idx = self._generate_frame_indices(
                    start_frame, end_frame, self.num_frames, self.sample_stride
                )
                # For output gt video
                video = vr.get_batch(frame_idx).permute(0, 3, 1, 2)
                # video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
                video = video[:, :, s_y:e_y, s_x:e_x]
                video = self._short_resize_and_crop(video, self.width, self.height)
                ####防止cross没有，那就还用regular
                frame_idx_regular = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["annotation"]["ann_frame_data"]["ann_frame_idx"]
                input_image = (
                    vr.get_batch([int(frame_idx_regular)]).cpu().numpy()[0].astype(np.uint8)
                )
                input_image = input_image[s_y:e_y, s_x:e_x]
                image_width = input_image.shape[1]
                image_height = input_image.shape[0]
                class_names = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["annotation"]["mask_map"]
                bbox_data = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["annotation"]["ann_frame_data"]["annotations"]
                mask_data = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["annotation"]["mask_annotation"][str(frame_idx_regular)]
                regular_subject_images = []
                subject_image_buffer = []
                ratio1 = []
                ratio2 = []
                for i, annotation_idx in enumerate(mask_data):
                    # try:
                    #     gme_score = bbox_data[int(annotation_idx) - 1]["gme_score"]
                    # except KeyError:
                    #     save_path = "/root/paddlejob/workspace/env_run/wanglei/debug_outputs/pro_data.json"
                    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #     with open(save_path, "w", encoding="utf-8") as f:
                    #         json.dump(data, f, ensure_ascii=False, indent=2)
                    #     print(f"[SAVE] Problematic data saved to: {save_path}")
                    #     assert 2==1

                    subject_image, class_name, gme_score, aes_score, crop_ratio = (
                        self.get_cropped_subject_image(
                            input_image,
                            annotation_idx,
                            class_names,
                            mask_data,
                            image_width,
                            image_height,
                            bbox_data,
                            use_bbox=False,
                        )
                    )
                    subject_image = self.crop_and_resize(subject_image, self.width, self.height)
                    subject_image_buffer.append(subject_image)
                    ratio2.append(crop_ratio)
                    if aes_score >=4.5 or gme_score >=0.6:
                        # print("regular",aes_score, gme_score, crop_ratio)
                        regular_subject_images.append(subject_image)
                        ratio1.append(crop_ratio)
                if len(regular_subject_images) ==0:#防止subject image为空
                    if len(subject_image_buffer) > 3:
                        top3_idx = sorted(range(len(ratio2)), key=lambda i: ratio2[i], reverse=True)[:3]
                        # 按索引取出对应图片
                        regular_subject_images = [subject_image_buffer[i] for i in top3_idx]
                    else:
                        regular_subject_images = subject_image_buffer
                if len(regular_subject_images) > 3:
                    # 取 ratio1 中值最大的前三个索引
                    top3_idx = sorted(range(len(ratio1)), key=lambda i: ratio1[i], reverse=True)[:3]
                    # 按索引取出对应图片
                    regular_subject_images = [regular_subject_images[i] for i in top3_idx]
                s2v_data["video"] = video #data #TCHW
                # print(s2v_data["video"].shape)
                del vr
                gc.collect()
                os.remove(local_path)
                #prompt
                s2v_data["prompt"] = data["first_frame_meta"]["opens2v_instances"][data["video_id"]]["metadata"]["face_cap_qwen"]
                # print(s2v_data)
                #subject image
                # s2v_data["subject_image"] = regular_subject_images
                # s2v_data["data_type"] = "regular"
                    # print("no cross")
                    # print("regu", len(regular_subject_images))
                # if not self._valid_sample(s2v_data):
                #     continue
                # yield s2v_data
                ########
                main_data_id = data["bos_path"].split("/")[4] + "/" + data["video_id"]
                # rng = random.Random(42)  # 局部随机器

                # selected_cross_data_id = rng.choice()
                # idx_cross_key = data["ref_meta"]["opens2v_annos"]["cluster_keys"].index(selected_cross_data_id)
                idx_cross_key = None
                selected_cross_data_id = None
                aft_class_ids = None
                # print(main_data_id)
                # print(data["opens2v_cross_data_info"])
                for id, cross_key in enumerate(data["ref_meta"]["opens2v_annos"]["cluster_keys"]):
                    cross_key = data["bos_path"].split("/")[4] + "/" + cross_key
                    filtered_cross_data = []
                    for temp_item in data["opens2v_cross_data_info"]:
                        if (
                            temp_item["cur_id"] == main_data_id
                            and temp_item["aft_id"] == cross_key
                        ) or (
                            temp_item["aft_id"] == main_data_id
                            and temp_item["cur_id"] == cross_key
                        ):
                            filtered_cross_data.append(temp_item)
                    aft_class = [
                        item["aft_class_id"] if item["cur_id"] == main_data_id else item["cur_class_id"]
                        for item in filtered_cross_data
                    ]
                    if len(aft_class) !=0:
                        aft_class_ids = aft_class
                        idx_cross_key = id
                        selected_cross_data_id = cross_key
                        break
                if aft_class_ids is None:
                    s2v_data["subject_image"] = regular_subject_images
                    s2v_data["data_type"] = "regular"
                    # print("no cross")
                    # print("regu", len(regular_subject_images))
                    if not self._valid_sample(s2v_data):
                        continue
                    yield s2v_data
                    continue
                ### cross video
                # print("cross")
                try:
                    local_path_cross = open_with_decord_from_url(data["ref_meta"]["opens2v_annos"]["cross_parts_bos_url"][idx_cross_key])
                    vr = VideoReader(local_path_cross, num_threads=4)
                except Exception as e:
                    print(f"[WARN] skip sample (video_id={data.get('video_id')}) due to: {e}")
                    continue
                # crop (remove watermark) & cut (remove transition)
                s_x, e_x, s_y, e_y = data["ref_meta"]["opens2v_annos"]["cluster_items"][idx_cross_key]["metadata"]["crop"]
                # print(s_x, e_x, s_y, e_y)
                
                frame_idx_cross = data["ref_meta"]["opens2v_annos"]["cluster_items"][idx_cross_key]["annotation"]["ann_frame_data"]["ann_frame_idx"]
                # print(frame_idx_cross)
                input_image = (
                    vr.get_batch([int(frame_idx_cross)]).cpu().numpy()[0].astype(np.uint8)
                )
                # print("cross_image",input_image.shape)
                input_image = input_image[s_y:e_y, s_x:e_x]
                # print(input_image.shape)
                image_width = input_image.shape[1]
                image_height = input_image.shape[0]
                # print("cross image",image_height, image_width)
                del vr
                gc.collect()
                os.remove(local_path_cross)
                class_names = data["ref_meta"]["opens2v_annos"]["cluster_items"][idx_cross_key]["annotation"]["mask_map"]
                bbox_data = data["ref_meta"]["opens2v_annos"]["cluster_items"][idx_cross_key]["annotation"]["ann_frame_data"]["annotations"]
                mask_data = data["ref_meta"]["opens2v_annos"]["cluster_items"][idx_cross_key]["annotation"]["mask_annotation"][str(frame_idx_cross)]
                subject_images = []
                # print(class_names)
                # print("aft_ids", aft_class_ids)
                subject_image_buffer_cross = []
                ratio1 = []
                ratio2 = []
                for i, annotation_idx in enumerate(mask_data):
                    # print(annotation_idx)
                    if annotation_idx not in aft_class_ids:
                        continue
                    # try:
                    #     gme_score = bbox_data[int(annotation_idx) - 1]["gme_score"]
                    # except KeyError:
                    #     save_path = "/root/paddlejob/workspace/env_run/wanglei/debug_outputs/pro_data_cross.json"
                    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #     with open(save_path, "w", encoding="utf-8") as f:
                    #         json.dump(data, f, ensure_ascii=False, indent=2)
                    #     print(f"[SAVE] Problematic data saved to: {save_path}")
                    #     assert 2==1
                    subject_image, class_name, gme_score, aes_score, crop_ratio = (
                        self.get_cropped_subject_image(
                            input_image,
                            annotation_idx,
                            class_names,
                            mask_data,
                            image_width,
                            image_height,
                            bbox_data,
                            use_bbox=False,
                        )
                    )
                    subject_image = self.crop_and_resize(subject_image, self.width, self.height)
                    subject_image_buffer_cross.append(subject_image)
                    ratio2.append(crop_ratio)
                    if aes_score >=4.5 or gme_score >=0.6:
                        # print("cross",aes_score, gme_score, crop_ratio)
                        subject_images.append(subject_image)
                        ratio1.append(crop_ratio)
                if len(subject_images) ==0:#防止cross subject image为空
                    if len(subject_image_buffer_cross) > 3:
                        top3_idx = sorted(range(len(ratio2)), key=lambda i: ratio2[i], reverse=True)[:3]
                        # 按索引取出对应图片
                        subject_images = [subject_image_buffer_cross[i] for i in top3_idx]
                    else:
                        subject_images = subject_image_buffer_cross
                if len(subject_images) > 3:
                    # 取 ratio1 中值最大的前三个索引
                    top3_idx = sorted(range(len(ratio1)), key=lambda i: ratio1[i], reverse=True)[:3]
                    # 按索引取出对应图片
                    subject_images = [subject_images[i] for i in top3_idx]
                # save_dir = "/root/paddlejob/workspace/env_run/wanglei/stream_subject_cross"  # 你要保存的目录
                # os.makedirs(save_dir, exist_ok=True)

                # for i, img in enumerate(subject_images):
                #     save_path = os.path.join(save_dir, f"subject_{i}.png")
                #     img.save(save_path)
                #     print(f"保存成功: {save_path}")
                # print("a data")
                # subject_images = torch.stack(subject_images, dim=0)
                # print(subject_images.shape) ##0.355 30%
                if random.random() < 0.23:
                    s2v_data["subject_image"] = subject_images
                    s2v_data["data_type"] = "cross"
                else:
                    s2v_data["subject_image"] = regular_subject_images
                    s2v_data["data_type"] = "regular"
                # print("cross", len(subject_images))
                # print(s2v_data)
                if not ok:
                    continue
                if not self._valid_sample(s2v_data):
                        continue
                yield s2v_data

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x,
    save_every_steps: int | None = None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.save_every_steps = save_every_steps  # 新增：按步数保存的间隔
    
    def on_step_end(self, accelerator, model, step_id):
        if self.save_every_steps and step_id % self.save_every_steps == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state_dict = accelerator.get_state_dict(model)
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
                state_dict = self.state_dict_converter(state_dict)
                os.makedirs(self.output_path, exist_ok=True)
                path = os.path.join(self.output_path, f"step-{step_id}.safetensors")
                accelerator.save(state_dict, path, safe_serialization=True)
    # def on_step_end(self, accelerator, model, step_id):
    #     if self.save_every_steps and step_id % self.save_every_steps == 0:
    #         accelerator.wait_for_everyone()
    #         if accelerator.is_main_process:
    #             unwrapped = accelerator.unwrap_model(model)

    #             # ✅ 只取整个 DiT 的权重
    #             dit = unwrapped.pipe.dit
    #             state_dict = accelerator.get_state_dict(dit)  # 等价于 dit.state_dict() + DDP 处理

    #             # 如果你还想去掉前缀，比如 "pipe.dit."
    #             if self.remove_prefix_in_ckpt:
    #                 new_sd = {}
    #                 prefix = self.remove_prefix_in_ckpt
    #                 for k, v in state_dict.items():
    #                     if k.startswith(prefix):
    #                         new_k = k[len(prefix):]
    #                     else:
    #                         new_k = k
    #                     new_sd[new_k] = v
    #                 state_dict = new_sd

    #             os.makedirs(self.output_path, exist_ok=True)
    #             path = os.path.join(self.output_path, f"dit-step-{step_id}.safetensors")
    #             accelerator.save(state_dict, path, safe_serialization=True)

    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
    # def on_epoch_end(self, accelerator, model, epoch_id):
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         unwrapped = accelerator.unwrap_model(model)
    #         dit = unwrapped.pipe.dit

    #         state_dict = accelerator.get_state_dict(dit)

    #         if self.remove_prefix_in_ckpt:
    #             new_sd = {}
    #             prefix = self.remove_prefix_in_ckpt
    #             for k, v in state_dict.items():
    #                 new_k = k[len(prefix):] if k.startswith(prefix) else k
    #                 new_sd[new_k] = v
    #             state_dict = new_sd

    #         os.makedirs(self.output_path, exist_ok=True)
    #         path = os.path.join(self.output_path, f"dit-epoch-{epoch_id}.safetensors")
    #         accelerator.save(state_dict, path, safe_serialization=True)


def count_records_with_ijson(path: str, json_mode: str) -> int:
    cnt = 0
    if json_mode == "array":
        with open(path, "r", encoding="utf-8") as f:
            for _ in ijson.items(f, "item"):
                cnt += 1
    elif json_mode == "kv":
        with open(path, "r", encoding="utf-8") as f:
            for _, rec in ijson.kvitems(f, ""):
                if isinstance(rec, dict):
                    cnt += 1
    elif json_mode == "multiv":
        with open(path, "r", encoding="utf-8") as f:
            for obj in ijson.items(f, "", multiple_values=True):
                if isinstance(obj, dict):
                    cnt += 1
    else:
        raise ValueError("json_mode must be one of: array, kv, multiv")
    return cnt


# def launch_training_task(
#     dataset: torch.utils.data.Dataset,
#     model: DiffusionTrainingModule,
#     model_logger: ModelLogger,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     num_epochs: int = 1,
#     gradient_accumulation_steps: int = 1,
# ):
  
#     # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=0)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=None,          # ← 逐样本直出：不会触发任何 collate/stack/cat
#         num_workers=0,
#         pin_memory=False,
#         persistent_workers=False
#     )
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         # split_batches=False,      # ❗不要在 batch 维度再拆
#         # device_placement=False,   # ❗不要递归把非张量（PIL）搬 device
#         # even_batches=False,     # 若版本支持，也建议关
#     )
#     model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
#     # try:
#     #     meta_path = getattr(dataset, "metadata_path", None)
#     #     json_mode = getattr(dataset, "json_mode", "array")
#     #     if meta_path is None:
#     #         raise AttributeError("'metadata_path' not found on dataset")
#     #     total_records = int(count_records_with_ijson(meta_path, json_mode))
#     #     if accelerator.is_main_process:
#     #         accelerator.print(f"[Dataset Info] Total records = {total_records}")
#     # except Exception as e:
#     #     if accelerator.is_main_process:
#     #         accelerator.print(f"[Warning] Failed to count dataset: {e}")
#     #     total_records = None  # 进度条走无限流模式
#     total_records = 32112
#     # 估算每个 rank 实际要迭代的步数（你的 IterableDataset 在 __iter__ 里做了 idx%world==rank 的切分）
#     world = accelerator.num_processes
#     if isinstance(total_records, int) and total_records > 0:
#         steps_per_epoch_rank = (total_records + world - 1) // world  # ceil
#         total_steps_rank = steps_per_epoch_rank * num_epochs
#     else:
#         steps_per_epoch_rank = None
#         total_steps_rank = None  # tqdm 无固定总数

#     # 进度条：若能拿到总数就设置 total，否则让它无限进度
#     if total_steps_rank is not None:
#         pbar = tqdm(total=total_steps_rank, disable=not accelerator.is_local_main_process, leave=True)
#     else:
#         pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
#     if accelerator.is_local_main_process:
#         pbar.set_description(f"Epoch 1/{num_epochs}")
#     # pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
#     global_step = 0
#     update_step = 0  # 新增：统计“优化器更新”的步数
#     for epoch_id in range(num_epochs):
#         if accelerator.is_local_main_process:
#             pbar.set_description(f"Epoch {epoch_id+1}/{num_epochs}")
#         for data in dataloader:
#             # print(data)
#             # print("[train] before forward")
#             with accelerator.accumulate(model):
#                 optimizer.zero_grad()
#                 loss, align_loss1, align_loss2 = model(data)
#                 # print("[train] after forward", float(loss.detach().cpu()))
#                 accelerator.backward(loss)
#                 optimizer.step()
#                 # 只有发生同步梯度（不是被 accumulate 屏蔽的 no_sync）才算一次“更新”
#                 if accelerator.sync_gradients:
#                     update_step += 1
#                     # 这里触发“每 N 步保存一次”
#                     model_logger.on_step_end(accelerator, model, update_step)
#                 # model_logger.on_step_end(loss)
#                 scheduler.step()
#             if accelerator.is_local_main_process:
#                 pbar.update(1)
#                 pbar.set_postfix({
#                 "loss": f"{loss.item():.4f}",
#                 "align1": f"{align_loss1.item():.4f}",
#                 "align2": f"{align_loss2.item():.4f}"
#                 })
#         model_logger.on_epoch_end(accelerator, model, epoch_id)
#     pbar.close()

import math

# def launch_training_task(
#     dataset: torch.utils.data.Dataset,
#     model: DiffusionTrainingModule,
#     model_logger: ModelLogger,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler.LRScheduler,
#     num_epochs: int = 1,
#     gradient_accumulation_steps: int = 1,
#     max_samples_per_epoch: int | None = 1000,   # ← 新增：每个 epoch 想看的“全局样本数”
# ):
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=None,          # 逐样本
#         num_workers=0,
#         pin_memory=False,
#         persistent_workers=False
#     )
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#     )
#     model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

#     # ------------ 计算每个 rank 的步数上限（确保全局≈1000 条） ------------
#     world = accelerator.num_processes
#     if max_samples_per_epoch is not None:
#         steps_per_epoch_rank = math.ceil(max_samples_per_epoch / max(1, world))
#         total_steps_rank = steps_per_epoch_rank * num_epochs
#     else:
#         steps_per_epoch_rank = None
#         total_steps_rank = None

#     # ------------ 进度条 ------------
#     if total_steps_rank is not None:
#         pbar = tqdm(total=total_steps_rank, disable=not accelerator.is_local_main_process, leave=True)
#     else:
#         pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
#     if accelerator.is_local_main_process:
#         pbar.set_description(f"Epoch 1/{num_epochs}")

#     update_step = 0
#     model.train()
#     for epoch_id in range(num_epochs):
#         if accelerator.is_local_main_process:
#             pbar.set_description(f"Epoch {epoch_id+1}/{num_epochs}")

#         step_in_epoch = 0  # ← 统计本 rank 的步数
#         for data in dataloader:
#             with accelerator.accumulate(model):
#                 loss, loss_mse, align_loss1,align_loss2 = model(data)
#                 # loss = model(data)
#                 accelerator.backward(loss)

#                 # 只有发生同步（真正的参数更新）时才做 step / scheduler
#                 if accelerator.sync_gradients:
#                     optimizer.step()
#                     scheduler.step()
#                     optimizer.zero_grad(set_to_none=True)  # 建议放更新后
#                     update_step += 1
#                     model_logger.on_step_end(accelerator, model, update_step)

#             step_in_epoch += 1
#             if accelerator.is_local_main_process:
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     "loss": f"{loss.item():.4f}",
#                     "align1": f"{align_loss1.item():.4f}",
#                     # "align2": f"{align_loss2:.4f}",
#                     "loss_mse": f"{loss_mse.item():.4f}"
#                 })

#             # ←—— 关键：到达本 rank 上限就结束这个 epoch
#             if steps_per_epoch_rank is not None and step_in_epoch >= steps_per_epoch_rank:
#                 break

#         model_logger.on_epoch_end(accelerator, model, epoch_id)

#     pbar.close()

from typing import List
from PIL import Image

def pad_subject_images_to_3(
    imgs: List[Image.Image],
    num_refs: int = 3,
    pad_color=(255, 255, 255),
) -> List[Image.Image]:
    """
    把单个样本的 subject_image list 补齐到 num_refs 张：
      - 如果 > num_refs：截断
      - 如果 1~(num_refs-1)：用“空白 PIL 图”补齐（保持同样大小）
      - 如果为空：直接报错（按你的 _valid_sample 理论上不会发生）
    """
    if len(imgs) == 0:
        raise ValueError("subject_image 为空，但 _valid_sample 已经保证至少一张，这里说明上游有问题。")

    imgs = list(imgs)  # 拷一份，避免原地修改

    # 已经够长了就截断
    if len(imgs) >= num_refs:
        return imgs[:num_refs]

    # 用第一张的尺寸构造“空白图”
    w, h = imgs[0].size
    while len(imgs) < num_refs:
        blank = Image.new("RGB", (w, h), color=pad_color)
        imgs.append(blank)

    return imgs



from typing import List, Dict, Any

def s2v_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch: List[sample_dict]，每个 sample 形如：
      {
        "video": List[PIL.Image],        # T 帧
        "prompt": str,
        "subject_image": List[PIL.Image],# 1~3 张
        "data_type": "regular" / "cross" # 可选
      }

    返回：
      {
        "video": List[List[PIL.Image]],          # [B][T]
        "prompt": List[str],                     # [B]
        "subject_image": List[List[PIL.Image]],  # [B][3]
        "data_type": List[str] (如果存在)        # [B]
      }
    """
    videos: List[List[Image.Image]] = []
    prompts: List[str] = []
    subject_images_batch: List[List[Image.Image]] = []
    data_types: List[str] = []

    for sample in batch:
        # 1. video：保持 List[PIL.Image]，外面再做 preprocess_video_batch
        videos.append(sample["video"])

        # 2. prompt：直接收集成 List[str]
        prompts.append(sample["prompt"])

        # 3. subject_image：先 pad 到 3 张，再存成 List[List[PIL.Image]]
        if "subject_image" not in sample or sample["subject_image"] is None:
            raise ValueError("sample 中缺少 subject_image，但 _valid_sample 应该已经过滤掉了。")
        padded = pad_subject_images_to_3(sample["subject_image"], num_refs=3)
        subject_images_batch.append(padded)

        # 4. 可选：data_type 也打包成 List[str]
        if "data_type" in sample:
            data_types.append(sample["data_type"])

    batch_dict: Dict[str, Any] = {
        "video": videos,                       # List[List[PIL.Image]]
        "prompt": prompts,                     # List[str]
        "subject_image": subject_images_batch, # List[List[PIL.Image]] (每个 3 张)
    }
    if len(data_types) == len(batch):
        batch_dict["data_type"] = data_types  # List[str]

    return batch_dict

def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    max_samples_per_epoch: int | None = 1000,   # ← 新增：每个 epoch 想看的“全局样本数”
):
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=None,          # 逐样本
    #     num_workers=0,
    #     pin_memory=False,
    #     persistent_workers=False
    # )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=s2v_collate_fn, num_workers=8)
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=False,   # 你必须关掉的开关
    )


    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_config=dataloader_config,
    )

    print("check nccl prepare")    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    # model_to_check = accelerator.unwrap_model(model)
    print("over")
    # # 2. 打印所有参数的 index, name, shape
    # if accelerator.is_local_main_process:
    #     param_list = []
    #     for idx, (name, p) in enumerate(model_to_check.named_parameters()):
    #         param_list.append({
    #             "idx": idx,
    #             "name": name,
    #             "shape": list(p.shape),  # tuple 也行，这里转 list 方便 json
    #         })

    #     # 建议用你训练的输出目录，比如 args.output_path
    #     save_dir = "/root/paddlejob/workspace/env_run/wanglei/"
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, "param_index_map.json")

    #     with open(save_path, "w", encoding="utf-8") as f:
    #         json.dump(param_list, f, ensure_ascii=False, indent=2)

    #     print(f"[rank0] 已保存参数 index 映射到: {save_path}")

    # ------------ 计算每个 rank 的步数上限（确保全局≈1000 条） ------------
    world = accelerator.num_processes
    per_rank_batch_size = dataloader.batch_size
    if max_samples_per_epoch is not None:
        steps_per_epoch_rank = math.ceil(max_samples_per_epoch / (max(1, world * per_rank_batch_size)))
        total_steps_rank = steps_per_epoch_rank * num_epochs
    else:
        steps_per_epoch_rank = None
        total_steps_rank = None

    # ------------ 进度条 ------------
    if total_steps_rank is not None:
        pbar = tqdm(total=total_steps_rank, disable=not accelerator.is_local_main_process, leave=True)
    else:
        pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
    if accelerator.is_local_main_process:
        pbar.set_description(f"Epoch 1/{num_epochs}")

    update_step = 0
    model.train()
    for epoch_id in range(num_epochs):
        if accelerator.is_local_main_process:
            pbar.set_description(f"Epoch {epoch_id+1}/{num_epochs}")

        step_in_epoch = 0  # ← 统计本 rank 的步数
        regular_local = 0
        cross_local = 0
        global_regular = 0   # 上一次同步后统计到的“全局 regular 数”
        global_cross = 0     # 上一次同步后统计到的“全局 cross 数”
        for data in dataloader:
            if isinstance(data, dict) and "data_type" in data:
                if data["data_type"] == "regular":
                    regular_local += 1
                elif data["data_type"] == "cross":
                    cross_local += 1
            with accelerator.accumulate(model):
                loss, loss_mse, align_loss1,align_loss2 = model(data)
                # loss = model(data)
                accelerator.backward(loss)
                # if accelerator.is_local_main_process:
                #     model_to_check = accelerator.unwrap_model(model)
                #     print("\n[DEBUG] 检查一次 grad=None 的可训练参数")
                #     for idx, (name, p) in enumerate(model_to_check.named_parameters()):
                #         if p.requires_grad and p.grad is None:
                #             print(f"UNUSED PARAM #{idx}: {name}  shape={tuple(p.shape)}", flush=True)
                #     print("[DEBUG] 检查结束\n", flush=True)

                # 只有发生同步（真正的参数更新）时才做 step / scheduler
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # 建议放更新后
                    update_step += 1
                    model_logger.on_step_end(accelerator, model, update_step)
                    local_counts = torch.tensor(
                        [[regular_local, cross_local]],          # 注意这里多了一层 []，shape: (1, 2)
                        device=accelerator.device,
                        dtype=torch.long,
                    )  # shape: (1, 2)

                    gathered = accelerator.gather(local_counts)  # shape: (world, 2)


                    if accelerator.is_local_main_process:
                        global_regular = int(gathered[:, 0].sum().item())
                        global_cross   = int(gathered[:, 1].sum().item())

            step_in_epoch += 1
            if accelerator.is_local_main_process:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "align1": f"{align_loss1.item():.4f}",
                    # "align2": f"{align_loss2:.4f}",
                    "loss_mse": f"{loss_mse.item():.4f}",
                    "regular": global_regular,
                    "cross": global_cross,
                })

            # ←—— 关键：到达本 rank 上限就结束这个 epoch
            if steps_per_epoch_rank is not None and step_in_epoch >= steps_per_epoch_rank:
                break

        model_logger.on_epoch_end(accelerator, model, epoch_id)

    pbar.close()


def launch_training_task_data(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
  
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,          # ← 逐样本直出：不会触发任何 collate/stack/cat
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # split_batches=False,      # ❗不要在 batch 维度再拆
        # device_placement=False,   # ❗不要递归把非张量（PIL）搬 device
        # even_batches=False,     # 若版本支持，也建议关
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    # try:
    #     meta_path = getattr(dataset, "metadata_path", None)
    #     json_mode = getattr(dataset, "json_mode", "array")
    #     if meta_path is None:
    #         raise AttributeError("'metadata_path' not found on dataset")
    #     total_records = int(count_records_with_ijson(meta_path, json_mode))
    #     if accelerator.is_main_process:
    #         accelerator.print(f"[Dataset Info] Total records = {total_records}")
    # except Exception as e:
    #     if accelerator.is_main_process:
    #         accelerator.print(f"[Warning] Failed to count dataset: {e}")
    #     total_records = None  # 进度条走无限流模式
    total_records = 32112
    # 估算每个 rank 实际要迭代的步数（你的 IterableDataset 在 __iter__ 里做了 idx%world==rank 的切分）
    world = accelerator.num_processes
    if isinstance(total_records, int) and total_records > 0:
        steps_per_epoch_rank = (total_records + world - 1) // world  # ceil
        total_steps_rank = steps_per_epoch_rank * num_epochs
    else:
        steps_per_epoch_rank = None
        total_steps_rank = None  # tqdm 无固定总数

    # 进度条：若能拿到总数就设置 total，否则让它无限进度
    if total_steps_rank is not None:
        pbar = tqdm(total=total_steps_rank, disable=not accelerator.is_local_main_process, leave=True)
    else:
        pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
    if accelerator.is_local_main_process:
        pbar.set_description(f"Epoch 1/{num_epochs}")
    # pbar = tqdm(disable=not accelerator.is_local_main_process, leave=True)
    global_step = 0
    update_step = 0  # 新增：统计“优化器更新”的步数
    for epoch_id in range(num_epochs):
        if accelerator.is_local_main_process:
            pbar.set_description(f"Epoch {epoch_id+1}/{num_epochs}")
        for data in dataloader:
            # pass
            # if accelerator.is_local_main_process:
                # print(data)
            # print("[train] before forward")
        #     with accelerator.accumulate(model):
        #         optimizer.zero_grad()
        #         loss = model(data)
        #         # print("[train] after forward", float(loss.detach().cpu()))
        #         accelerator.backward(loss)
        #         optimizer.step()
        #         # 只有发生同步梯度（不是被 accumulate 屏蔽的 no_sync）才算一次“更新”
        #         if accelerator.sync_gradients:
        #             update_step += 1
        #             # 这里触发“每 N 步保存一次”
        #             model_logger.on_step_end(accelerator, model, update_step)
        #         # model_logger.on_step_end(loss)
        #         scheduler.step()
            if accelerator.is_local_main_process:
                pbar.update(1)
        # model_logger.on_epoch_end(accelerator, model, epoch_id)
    pbar.close()


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        # print(data)
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            # torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        nargs="+",                 # <--- 新增
        default=None,
        help="Path(s) to metadata json files. Support multiple paths.",
    )
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--json_mode", type=str, default="kv", help="Json mode.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser