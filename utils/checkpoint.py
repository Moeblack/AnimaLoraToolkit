"""
Checkpoint Manager Module - Checkpoint 管理
==========================================
处理模型 checkpoint 的保存和加载

功能：
1. 定期保存 LoRA 权重
2. 保存完整训练状态（优化器、学习率调度器、随机种子等）
3. 支持从 checkpoint 恢复训练
4. 自动管理 checkpoint 数量（防止磁盘占满）
5. 保存最佳模型

Checkpoint 结构：
```
checkpoint-1000/
├── lora_weights.safetensors  # LoRA 权重
├── optimizer_state.pt        # 优化器状态（可选）
├── scheduler_state.pt        # 学习率调度器状态（可选）
├── random_states.pkl         # 随机种子状态（可选）
└── training_state.json       # 训练状态元数据
```
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
from torch import nn
from torch.optim import Optimizer

from accelerate import Accelerator
from accelerate.logging import get_logger

from diffusers.training_utils import EMAModel
from safetensors.torch import save_file, load_file

logger = get_logger(__name__)


class CheckpointManager:
    """
    Checkpoint 管理器
    
    负责：
    - 创建和管理 checkpoint 目录结构
    - 保存和加载模型权重
    - 保存和加载训练状态（优化器、调度器等）
    - 自动清理旧 checkpoint
    """
    
    def __init__(
        self,
        output_dir: str,
        total_limit: Optional[int] = None,
        save_state: bool = True,
    ):
        """
        初始化 Checkpoint 管理器
        
        Args:
            output_dir: checkpoint 输出目录
            total_limit: 保留的最大 checkpoint 数量（None 表示不限制）
            save_state: 是否保存完整训练状态
        """
        self.output_dir = Path(output_dir)
        self.total_limit = total_limit
        self.save_state = save_state
        
        # 创建 checkpoint 目录
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 跟踪已保存的 checkpoints
        self.saved_checkpoints = []
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """扫描已存在的 checkpoints"""
        if not self.checkpoint_dir.exists():
            return
        
        for checkpoint_path in sorted(self.checkpoint_dir.glob("checkpoint-*")):
            if checkpoint_path.is_dir():
                try:
                    step = int(checkpoint_path.name.split("-")[-1])
                    self.saved_checkpoints.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        # 按步数排序
        self.saved_checkpoints.sort(key=lambda x: x[0])
        
        if self.saved_checkpoints:
            logger.info(f"Found {len(self.saved_checkpoints)} existing checkpoints")
    
    def save_checkpoint(
        self,
        accelerator: Accelerator,
        transformer: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        global_step: int = 0,
        epoch: int = 0,
        ema_model: Optional[EMAModel] = None,
        additional_state: Optional[Dict[str, Any]] = None,
    ):
        """
        保存 checkpoint
        
        Args:
            accelerator: Accelerator 实例
            transformer: Transformer 模型（带 LoRA）
            optimizer: 优化器（可选）
            lr_scheduler: 学习率调度器（可选）
            global_step: 当前全局步数
            epoch: 当前 epoch
            ema_model: EMA 模型（可选）
            additional_state: 额外的状态字典（可选）
        """
        # 只在主进程保存
        if not accelerator.is_main_process:
            return
        
        # 创建 checkpoint 目录
        checkpoint_name = f"checkpoint-{global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # ----------------------------------------------------------------------
        # 1. 保存 LoRA 权重
        # ----------------------------------------------------------------------
        # 解包装模型（去除 DDP wrapper）
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        
        # 保存 LoRA 权重
        lora_weights_path = checkpoint_path / "lora_weights.safetensors"
        
        # 获取 LoRA 状态字典
        from peft.utils import get_peft_model_state_dict
        lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
        
        # 使用 safetensors 保存（更安全、更快）
        save_file(lora_state_dict, str(lora_weights_path))
        logger.info(f"  ✓ LoRA weights saved: {lora_weights_path}")
        
        # ----------------------------------------------------------------------
        # 2. 保存训练状态（可选）
        # ----------------------------------------------------------------------
        if self.save_state:
            # 保存优化器状态
            if optimizer is not None:
                optimizer_path = checkpoint_path / "optimizer_state.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
                logger.info(f"  ✓ Optimizer state saved")
            
            # 保存学习率调度器状态
            if lr_scheduler is not None:
                scheduler_path = checkpoint_path / "scheduler_state.pt"
                torch.save(lr_scheduler.state_dict(), scheduler_path)
                logger.info(f"  ✓ Scheduler state saved")
            
            # 保存 EMA 模型
            if ema_model is not None:
                ema_path = checkpoint_path / "ema_model.pt"
                torch.save(ema_model.state_dict(), ema_path)
                logger.info(f"  ✓ EMA model saved")
            
            # 保存随机种子状态
            random_states_path = checkpoint_path / "random_states.pkl"
            random_states = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": None,  # 可以添加 python random 状态
            }
            with open(random_states_path, "wb") as f:
                pickle.dump(random_states, f)
            logger.info(f"  ✓ Random states saved")
        
        # ----------------------------------------------------------------------
        # 3. 保存训练状态元数据
        # ----------------------------------------------------------------------
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "checkpoint_time": datetime.now().isoformat(),
            "save_state": self.save_state,
        }
        
        # 合并额外状态
        if additional_state:
            training_state.update(additional_state)
        
        training_state_path = checkpoint_path / "training_state.json"
        with open(training_state_path, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"  ✓ Training state saved")
        
        # ----------------------------------------------------------------------
        # 4. 更新 checkpoint 列表并清理旧 checkpoint
        # ----------------------------------------------------------------------
        self.saved_checkpoints.append((global_step, checkpoint_path))
        self.saved_checkpoints.sort(key=lambda x: x[0])
        
        if self.total_limit is not None:
            self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint {checkpoint_name} saved successfully!")
    
    def _cleanup_old_checkpoints(self):
        """清理旧的 checkpoints，只保留最新的 N 个"""
        while len(self.saved_checkpoints) > self.total_limit:
            # 删除最早的 checkpoint
            step, checkpoint_path = self.saved_checkpoints.pop(0)
            
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(f"  Removed old checkpoint: {checkpoint_path.name}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        accelerator: Accelerator,
        logger=None,
    ) -> Tuple[int, int]:
        """
        从 checkpoint 恢复训练
        
        Args:
            checkpoint_path: checkpoint 路径或步数（如 "checkpoint-1000" 或 "1000"）
            accelerator: Accelerator 实例
            logger: 日志记录器
            
        Returns:
            Tuple[int, int]: (global_step, epoch)
            
        Raises:
            FileNotFoundError: 如果 checkpoint 不存在
        """
        # 解析 checkpoint 路径
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.is_absolute():
            # 如果提供的是步数，构建完整路径
            if checkpoint_path.name.isdigit():
                checkpoint_path = self.checkpoint_dir / f"checkpoint-{checkpoint_path.name}"
            else:
                checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if logger:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # ----------------------------------------------------------------------
        # 1. 加载训练状态元数据
        # ----------------------------------------------------------------------
        training_state_path = checkpoint_path / "training_state.json"
        if training_state_path.exists():
            with open(training_state_path, "r", encoding="utf-8") as f:
                training_state = json.load(f)
            
            global_step = training_state.get("global_step", 0)
            epoch = training_state.get("epoch", 0)
            save_state = training_state.get("save_state", False)
        else:
            # 尝试从目录名解析步数
            try:
                global_step = int(checkpoint_path.name.split("-")[-1])
            except ValueError:
                global_step = 0
            epoch = 0
            save_state = False
        
        if logger:
            logger.info(f"  Resuming from step {global_step}, epoch {epoch}")
        
        # ----------------------------------------------------------------------
        # 2. 加载 LoRA 权重
        # ----------------------------------------------------------------------
        lora_weights_path = checkpoint_path / "lora_weights.safetensors"
        if lora_weights_path.exists():
            lora_state_dict = load_file(str(lora_weights_path))
            
            # 加载到模型
            unwrapped_transformer = accelerator.unwrap_model(accelerator.transformer if hasattr(accelerator, 'transformer') else None)
            if unwrapped_transformer is not None:
                # 使用 strict=False 允许部分加载
                unwrapped_transformer.load_state_dict(lora_state_dict, strict=False)
                if logger:
                    logger.info(f"  ✓ LoRA weights loaded")
        else:
            if logger:
                logger.warning(f"  ⚠ LoRA weights not found at {lora_weights_path}")
        
        # ----------------------------------------------------------------------
        # 3. 加载训练状态（如果存在）
        # ----------------------------------------------------------------------
        if save_state and self.save_state:
            # 加载优化器状态
            optimizer_path = checkpoint_path / "optimizer_state.pt"
            if optimizer_path.exists():
                # 注意：优化器状态需要在 optimizer 创建后加载
                # 这里返回路径，由调用者处理
                if logger:
                    logger.info(f"  ✓ Optimizer state found (load manually)")
            
            # 加载调度器状态
            scheduler_path = checkpoint_path / "scheduler_state.pt"
            if scheduler_path.exists():
                if logger:
                    logger.info(f"  ✓ Scheduler state found (load manually)")
            
            # 加载随机状态
            random_states_path = checkpoint_path / "random_states.pkl"
            if random_states_path.exists():
                with open(random_states_path, "rb") as f:
                    random_states = pickle.load(f)
                
                # 恢复随机状态
                torch.set_rng_state(random_states["torch"])
                if random_states["cuda"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(random_states["cuda"])
                
                if logger:
                    logger.info(f"  ✓ Random states restored")
        
        if logger:
            logger.info(f"Checkpoint loaded successfully!")
        
        return global_step, epoch
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        获取最新的 checkpoint 路径
        
        Returns:
            Optional[Path]: 最新 checkpoint 路径，如果没有则返回 None
        """
        if not self.saved_checkpoints:
            return None
        
        return self.saved_checkpoints[-1][1]
    
    def list_checkpoints(self) -> list:
        """
        列出所有可用的 checkpoints
        
        Returns:
            list: checkpoint 路径列表
        """
        return [path for _, path in self.saved_checkpoints]


def save_final_model(
    output_dir: str,
    accelerator: Accelerator,
    transformer: nn.Module,
    save_format: str = "safetensors",
):
    """
    保存最终模型
    
    Args:
        output_dir: 输出目录
        accelerator: Accelerator 实例
        transformer: Transformer 模型
        save_format: 保存格式 ("safetensors" 或 "pytorch")
    """
    if not accelerator.is_main_process:
        return
    
    final_dir = Path(output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # 解包装模型
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    
    # 获取 LoRA 状态字典
    from peft.utils import get_peft_model_state_dict
    lora_state_dict = get_peft_model_state_dict(unwrapped_transformer)
    
    # 保存
    if save_format == "safetensors":
        output_path = final_dir / "model.safetensors"
        save_file(lora_state_dict, str(output_path))
    else:
        output_path = final_dir / "model.pt"
        torch.save(lora_state_dict, output_path)
    
    logger.info(f"Final model saved to {output_path}")
