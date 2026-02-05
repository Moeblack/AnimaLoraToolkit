"""
Dataset Module - 标签数据集处理
===============================
处理 Anima 风格的 tag-based 数据集
每个图片对应一个同名的 .txt 文件，包含 Danbooru 风格的标签

为什么使用 Dataset 类：
1. 统一数据加载接口
2. 支持数据预处理和增强
3. 兼容 PyTorch DataLoader
4. 支持缓存和惰性加载
"""

import os
import random
import bisect
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps
import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


class TagBasedDataset(Dataset):
    """
    Tag-based 数据集类
    
    数据集结构：
    data_root/
        ├── image1.jpg
        ├── image1.txt
        ├── image2.png
        ├── image2.txt
        └── ...
    
    .txt 文件格式（Danbooru 风格）：
    1girl, oomuro sakurako, yuru yuri, brown hair, long hair, smile, ...
    
    特性：
    - 支持多种图像格式（jpg, png, webp, jxl 等）
    - 支持 tag dropout（随机丢弃标签以增强泛化）
    - 支持图像数据增强（随机裁剪、翻转等）
    - 支持多分辨率训练
    """
    
    # 支持的图像扩展名
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.jxl'}
    
    def __init__(
        self,
        data_root: str,
        tokenizer,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        tag_dropout: float = 0.1,
        max_length: int = 77,
        enable_arb: bool = False,
        min_bucket_reso: int = 512,
        max_bucket_reso: int = 1024,
        bucket_reso_steps: int = 64,
        bucket_no_upscale: bool = False,
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据集根目录路径
            tokenizer: 文本编码器的 tokenizer
            resolution: 训练图像分辨率（正方形）
            center_crop: 是否中心裁剪图像
            random_flip: 是否随机水平翻转
            tag_dropout: 标签丢弃概率（0-1）
            max_length: 最大 token 长度
        """
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tag_dropout = tag_dropout
        self.max_length = max_length

        # ARB (aspect ratio bucketing)
        self.enable_arb = bool(enable_arb)
        self.min_bucket_reso = int(min_bucket_reso)
        self.max_bucket_reso = int(max_bucket_reso)
        self.bucket_reso_steps = int(bucket_reso_steps)
        self.bucket_no_upscale = bool(bucket_no_upscale)
        
        # ------------------------------------------------------------------
        # 扫描数据集
        # ------------------------------------------------------------------
        # 找到所有有效的图像-标签对
        self.image_paths = []
        self.caption_paths = []
        self._entries = []
        self._cumulative_repeats = []
        
        self._scan_dataset()
        
        if len(self._entries) == 0:
            raise ValueError(
                f"在 {data_root} 中没有找到有效的图像-标签对。\n"
                f"请确保：\n"
                f"1. 目录中存在图片文件（支持的格式：{self.SUPPORTED_EXTENSIONS}）\n"
                f"2. 每个图片都有同名的 .txt 标签文件\n"
                f"例如：image.jpg 和 image.txt"
            )
        
        if self.enable_arb:
            self._setup_buckets()

        total_samples = self._build_repeat_index()
        if self.enable_arb:
            print(
                f"Found {len(self._entries)} training samples (with repeats: {total_samples}), "
                f"ARB buckets: {len(self._bucket_resolutions)}"
            )
        else:
            print(f"Found {len(self._entries)} training samples (with repeats: {total_samples})")
        
        # ------------------------------------------------------------------
        # 设置图像变换
        # ------------------------------------------------------------------
        # 变换流程：
        # 1. 调整大小（保持长宽比）
        # 2. 裁剪（中心或随机）
        # 3. 随机翻转（数据增强）
        # 4. 转换为张量
        # 5. 归一化到 [-1, 1]
        
        self.transforms = self._build_transforms()

        # Post transforms are always used; in ARB mode we handle resize/crop per sample.
        self._post_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def _scan_dataset(self):
        """
        扫描数据集目录，找到所有有效的图像-标签对
        
        为什么需要扫描：
        1. 验证数据完整性
        2. 提前发现缺失的标签文件
        3. 构建索引，加速后续访问
        """
        def add_entry(image_path: Path, caption_path: Path, repeats: int) -> None:
            self.image_paths.append(image_path)
            self.caption_paths.append(caption_path)
            self._entries.append({
                "image_path": image_path,
                "caption_path": caption_path,
                "repeats": repeats,
            })

        def collect_pairs(base_dir: Path, repeats: int) -> None:
            for ext in self.SUPPORTED_EXTENSIONS:
                for image_path in base_dir.glob(f"*{ext}"):
                    caption_path = image_path.with_suffix('.txt')
                    if caption_path.exists():
                        add_entry(image_path, caption_path, repeats)
                        continue
                    alt_caption_path = image_path.with_suffix('.caption')
                    if alt_caption_path.exists():
                        add_entry(image_path, alt_caption_path, repeats)

        # 1) files directly under data_root (repeat=1)
        collect_pairs(self.data_root, 1)

        # 2) subdirectories with optional repeats prefix, e.g. 10_lance
        for subdir in self.data_root.iterdir():
            if not subdir.is_dir():
                continue
            repeats = self._parse_repeats_from_dir(subdir.name)
            if repeats < 1:
                print(f"Warning: ignoring subdirectory (invalid repeat count): {subdir.name}")
                continue
            collect_pairs(subdir, repeats)

    def _parse_repeats_from_dir(self, name: str) -> int:
        prefix = name.split("_", 1)[0]
        if not prefix.isdigit():
            return 1
        try:
            return max(int(prefix), 0)
        except ValueError:
            return 1

    def _build_repeat_index(self) -> int:
        self._cumulative_repeats = []
        total = 0
        for entry in self._entries:
            repeats = max(int(entry.get("repeats", 1)), 1)
            total += repeats
            self._cumulative_repeats.append(total)
        return total

    def _get_entry_index(self, idx: int) -> int:
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        # Use bisect_right to find the correct entry for repeated samples
        # For cumulative [3, 6, 9], idx=0,1,2 -> entry 0; idx=3,4,5 -> entry 1
        return bisect.bisect_right(self._cumulative_repeats, idx)
    
    def _build_transforms(self):
        """
        构建图像变换管道
        
        返回:
            transforms.Compose: 变换管道
        """
        transform_list = []
        
        # 第一步：调整大小
        # 使用 LANCZOS 插值算法，保持高质量
        if self.center_crop:
            # 中心裁剪模式：先将短边调整为 resolution
            transform_list.append(
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.LANCZOS
                )
            )
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            # 随机裁剪模式：先将长边调整为 resolution
            transform_list.append(
                transforms.Resize(
                    self.resolution,
                    interpolation=transforms.InterpolationMode.LANCZOS
                )
            )
            transform_list.append(transforms.RandomCrop(self.resolution))
        
        # 第二步：随机水平翻转（数据增强）
        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # 第三步：转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 第四步：归一化到 [-1, 1]
        # VAE 期望输入范围是 [-1, 1]，而不是 [0, 1]
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        
        return transforms.Compose(transform_list)

    def _generate_bucket_resolutions(self) -> List[Tuple[int, int]]:
        """Generate bucket resolutions (width, height) for ARB.

        Strategy:
        - Constrain both width/height to [min_bucket_reso, max_bucket_reso]
        - Constrain area to <= max_bucket_reso ** 2 (kohya-style)
        - Align to `bucket_reso_steps` (and to 8 at minimum)
        """
        step = max(int(self.bucket_reso_steps), 8)
        if step % 8 != 0:
            # align to 8 to keep VAE/patching constraints sane
            step = int(math.ceil(step / 8) * 8)

        min_reso = max(int(self.min_bucket_reso), step)
        max_reso = max(int(self.max_bucket_reso), min_reso)

        base = int(self.resolution)
        max_area = max_reso * max_reso

        buckets: set[Tuple[int, int]] = set()
        for w in range(min_reso, max_reso + 1, step):
            # largest possible h for this w under max_area, capped to max_reso
            h = int((max_area // max(w, 1) // step) * step)
            h = min(h, max_reso)
            if h < min_reso:
                continue
            # enforce area constraint
            if w * h > max_area:
                continue
            buckets.add((w, h))
            buckets.add((h, w))

        buckets.add((min(base, max_reso), min(base, max_reso)))
        return sorted(buckets, key=lambda x: (x[0] * x[1], x[0], x[1]))

    def _setup_buckets(self) -> None:
        self._bucket_resolutions: List[Tuple[int, int]] = self._generate_bucket_resolutions()
        if not self._bucket_resolutions:
            self._bucket_resolutions = [(self.resolution, self.resolution)]

        bucket_ratios = [w / h for w, h in self._bucket_resolutions]

        for entry in self._entries:
            try:
                with Image.open(entry["image_path"]) as im:
                    w0, h0 = im.size
            except Exception:
                w0 = self.resolution
                h0 = self.resolution

            if h0 <= 0:
                h0 = 1
            r0 = w0 / h0

            best_i = 0
            best_score = float("inf")
            for i, r in enumerate(bucket_ratios):
                bw, bh = self._bucket_resolutions[i]
                if self.bucket_no_upscale and max(bw, bh) > max(w0, h0):
                    continue

                # compare in log-space to treat reciprocal ratios symmetrically
                score = abs(math.log(r0) - math.log(r))
                if score < best_score:
                    best_score = score
                    best_i = i

            entry["bucket_id"] = best_i
            entry["bucket_reso"] = self._bucket_resolutions[best_i]
    
    def __len__(self):
        """返回数据集大小"""
        if not self._cumulative_repeats:
            return 0
        return self._cumulative_repeats[-1]
    
    def _load_caption(self, caption_path: Path) -> str:
        """
        加载并处理标签文件
        
        处理流程：
        1. 读取文件内容
        2. 处理 tag dropout（随机丢弃部分标签）
        3. 清理格式
        
        Args:
            caption_path: 标签文件路径
            
        Returns:
            str: 处理后的标签字符串
        """
        # 读取标签
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        # 如果启用 tag dropout，随机丢弃部分标签
        if self.tag_dropout > 0 and caption:
            tags = [tag.strip() for tag in caption.split(',')]
            
            # 保留至少一个标签（避免空提示）
            if len(tags) > 1:
                # 随机丢弃标签
                kept_tags = [
                    tag for tag in tags
                    if random.random() > self.tag_dropout
                ]
                
                # 确保至少保留一个标签
                if len(kept_tags) == 0:
                    kept_tags = [random.choice(tags)]
                
                caption = ', '.join(kept_tags)
        
        return caption
    
    def _tokenize_caption(self, caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 tokenizer 编码标签
        
        Args:
            caption: 标签字符串
            
        Returns:
            torch.Tensor: token IDs
        """
        # 编码标签
        # padding="max_length" 确保所有序列长度一致
        # truncation=True 截断超长序列
        inputs = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0)
    
    def _load_image(self, image_path: Path) -> Image.Image:
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: failed to load image {image_path}, error: {e}")
            return Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))

    def _resize_crop_to_bucket(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Resize preserving aspect ratio, then crop to (target_w, target_h)."""
        w, h = image.size
        if w <= 0 or h <= 0:
            return Image.new('RGB', (target_w, target_h), (128, 128, 128))

        scale = max(target_w / w, target_h / h)
        new_w = max(int(round(w * scale)), target_w)
        new_h = max(int(round(h * scale)), target_h)
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

        max_left = max(new_w - target_w, 0)
        max_top = max(new_h - target_h, 0)
        if self.center_crop:
            left = max_left // 2
            top = max_top // 2
        else:
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0

        return image.crop((left, top, left + target_w, top + target_h))

    def _get_processed_pixel_values(self, entry: Dict[str, Any]) -> torch.Tensor:
        image = self._load_image(entry["image_path"])

        if self.enable_arb and "bucket_reso" in entry:
            target_w, target_h = entry["bucket_reso"]
            image = self._resize_crop_to_bucket(image, int(target_w), int(target_h))
        else:
            image = self.transforms(image)
            # `self.transforms` already returns tensor
            return image

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)

        return self._post_transforms(image)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict: 包含以下键的字典
                - pixel_values: 图像张量 [3, H, W]
                - input_ids: token IDs [max_length]
                - caption: 原始标签字符串（用于调试）
        """
        # 加载图像
        entry_idx = self._get_entry_index(idx)
        entry = self._entries[entry_idx]

        # 应用变换
        pixel_values = self._get_processed_pixel_values(entry)
        
        # 加载标签
        caption_path = entry["caption_path"]
        caption = self._load_caption(caption_path)
        
        # 编码标签
        input_ids, attention_mask = self._tokenize_caption(caption)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,  # 用于调试和日志
        }


class CachedTagBasedDataset(TagBasedDataset):
    """
    带缓存的 TagBasedDataset
    
    特性：
    - 缓存预处理后的 latent（节省 VAE 编码时间）
    - 缓存 tokenized caption
    - 首次加载较慢，后续加载快
    
    适用场景：
    - 数据集不太大（可以放入内存）
    - 需要多次 epoch 训练
    - 想要最大化 GPU 利用率
    """
    
    def __init__(
        self,
        data_root: str,
        tokenizer,
        vae,
        device,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        tag_dropout: float = 0.1,
        max_length: int = 77,
        cache_dir: Optional[str] = None,
        enable_arb: bool = False,
        min_bucket_reso: int = 512,
        max_bucket_reso: int = 1024,
        bucket_reso_steps: int = 64,
        bucket_no_upscale: bool = False,
    ):
        """
        初始化缓存数据集
        
        Args:
            vae: VAE 模型，用于预编码图像到 latent
            device: 计算设备
            cache_dir: 缓存目录，如果为 None 则使用内存缓存
        """
        super().__init__(
            data_root=data_root,
            tokenizer=tokenizer,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            tag_dropout=tag_dropout,
            max_length=max_length,
            enable_arb=enable_arb,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_steps=bucket_reso_steps,
            bucket_no_upscale=bucket_no_upscale,
        )
        
        self.vae = vae
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 内存缓存
        self.latents_cache = {}
        self.tokens_cache = {}
        self.attn_mask_cache = {}
        
        # 如果使用磁盘缓存，创建目录
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
        
        # 预计算所有样本
        self._precompute_cache()
    
    def _load_disk_cache(self):
        """从磁盘加载缓存（如果存在）"""
        # TODO: 实现磁盘缓存加载
        pass
    
    def _precompute_cache(self):
        """
        预计算所有样本的 latent 和 tokens

        警告：这会消耗大量内存，确保数据集不会太大
        """
        print("Precomputing cache...")

        self.vae.eval()
        # Use VAE's dtype to avoid type mismatch
        vae_dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            for entry_idx, entry in enumerate(tqdm(self._entries, desc="Caching latents")):
                pixel_values = self._get_processed_pixel_values(entry).unsqueeze(0).to(self.device, dtype=vae_dtype)
                # QwenImage VAE expects [B, C, T, H, W] for images too.
                pixel_values = pixel_values.unsqueeze(2)
                latent = self.vae.encode(pixel_values).latent_dist.sample()
                latent = latent.squeeze(0).cpu()

                caption = self._load_caption(entry["caption_path"])
                input_ids, attention_mask = self._tokenize_caption(caption)

                self.latents_cache[entry_idx] = latent
                self.tokens_cache[entry_idx] = input_ids
                self.attn_mask_cache[entry_idx] = attention_mask

        print(f"Cache complete! Memory usage: ~{self._estimate_cache_size():.2f} GB")
    
    def _estimate_cache_size(self) -> float:
        """估计缓存占用的内存大小（GB）"""
        if len(self.latents_cache) == 0:
            return 0.0
        
        # 计算一个 latent 的大小
        sample_latent = next(iter(self.latents_cache.values()))
        latent_size = sample_latent.numel() * sample_latent.element_size()
        
        # 计算总大小
        total_bytes = latent_size * len(self.latents_cache)
        
        # 加上 tokens 的大小（相对很小）
        sample_tokens = next(iter(self.tokens_cache.values()))
        tokens_size = sample_tokens.numel() * sample_tokens.element_size()
        total_bytes += tokens_size * len(self.tokens_cache)
        
        return total_bytes / (1024 ** 3)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取缓存的数据"""
        # 应用随机翻转（必须在缓存后应用，因为翻转是随机的）
        entry_idx = self._get_entry_index(idx)
        latent = self.latents_cache[entry_idx].clone()
        
        if self.random_flip and random.random() < 0.5:
            # 水平翻转 latent
            latent = torch.flip(latent, dims=[-1])
        
        # 应用 tag dropout
        caption_path = self._entries[entry_idx]["caption_path"]
        caption = self._load_caption(caption_path)
        input_ids, attention_mask = self._tokenize_caption(caption)
        
        return {
            "latents": latent,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
        }


def collate_fn(examples):
    """
    DataLoader 的 collate 函数
    
    将一批样本组合成 batch tensor
    
    Args:
        examples: 样本列表，每个样本是 __getitem__ 返回的字典
        
    Returns:
        Dict: batch 数据
    """
    # 堆叠图像
    if "pixel_values" in examples[0]:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        pixel_values = None
    
    # 堆叠 latent（如果使用缓存数据集）
    if "latents" in examples[0]:
        latents = torch.stack([example["latents"] for example in examples])
        latents = latents.to(memory_format=torch.contiguous_format).float()
    else:
        latents = None
    
    # 堆叠 token IDs
    input_ids = torch.stack([example["input_ids"] for example in examples])

    # attention mask
    attention_mask = None
    if "attention_mask" in examples[0]:
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "latents": latents,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class BucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups indices by bucket id.

    This requires the dataset to expose `_entries` with `bucket_id` and the repeat index.
    """

    def __init__(
        self,
        dataset: TagBasedDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # best-effort length computation
        total = 0
        buckets = defaultdict(int)
        for entry in self.dataset._entries:
            bid = int(entry.get("bucket_id", 0))
            buckets[bid] += int(entry.get("repeats", 1))

        for count in buckets.values():
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        buckets: Dict[int, List[int]] = defaultdict(list)
        # Build sample indices per bucket, respecting repeats without bisect on each sample.
        start = 0
        for entry in self.dataset._entries:
            repeats = max(int(entry.get("repeats", 1)), 1)
            bid = int(entry.get("bucket_id", 0))
            buckets[bid].extend(range(start, start + repeats))
            start += repeats

        bucket_ids = list(buckets.keys())
        if self.shuffle:
            rng.shuffle(bucket_ids)
            for bid in bucket_ids:
                rng.shuffle(buckets[bid])

        for bid in bucket_ids:
            indices = buckets[bid]
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
