# AnimaLoraToolkit

一个功能完善的 **Anima** LoRA/LoKr 训练工具包，支持 YAML 配置、JSON 标签、实时监控，输出兼容 ComfyUI。

## 🔗 相关项目

训练好的 LoRA 可以在 ComfyUI 中使用，推荐搭配：

- **[ComfyUI-AnimaTool](https://github.com/Moeblack/ComfyUI-AnimaTool)** - Anima 图像生成工具，支持 MCP Server、HTTP API、CLI，可直接加载本工具训练的 LoRA

### 示例作品

使用本工具训练的 LoRA 示例：

- **[Cosmic Princess Kaguya | 超时空辉耀姬](https://civitai.com/models/2366705)** - 基于 Netflix 动画电影《超时空辉耀姬！》训练的画风+角色 LoKr

## 📦 安装

```bash
git clone https://github.com/Moeblack/AnimaLoraToolkit.git
cd AnimaLoraToolkit

# 创建虚拟环境
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

> 说明：`requirements.txt` 默认**不强制安装** `xformers / flash-attn / bitsandbytes / wandb`（可选项，很多环境尤其 Windows 会安装失败）。
> 需要时可按需 `pip install xformers`，并在配置里把 `xformers: true` 打开。

## 🚀 快速开始

### 1. 准备模型文件

```
models/
├── transformers/
│   └── anima-preview.safetensors      # Anima 主模型
├── vae/
│   └── qwen_image_vae.safetensors     # VAE 解码器
└── text_encoders/
    ├── config.json                     # 已包含（小文件）
    ├── tokenizer_config.json           # 已包含（小文件）
    ├── merges.txt                      # 已包含（小文件）
    ├── tokenizer.json                  # 需下载（小文件，可用 download_tokenizers.py）
    ├── vocab.json                      # 需下载（小文件，可用 download_tokenizers.py）
    ├── special_tokens_map.json         # 需下载（小文件，可用 download_tokenizers.py）
    └── model.safetensors               # 需下载（大文件：Qwen3-0.6B 权重）
```

**方式一：一键下载 tokenizer（推荐）**

```bash
python download_tokenizers.py
```

> 使用 hf-mirror.com 镜像，国内可直接访问。如需自定义镜像，设置环境变量 `HF_ENDPOINT`。

**方式二：手动下载**

| 文件 | 来源 | 放置位置 |
|------|------|----------|
| `anima-preview.safetensors` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/transformers/` |
| `qwen_image_vae.safetensors` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | `models/vae/` |
| `model.safetensors` | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` |
| `tokenizer.json`, `vocab.json` | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | `models/text_encoders/` |
| `spiece.model` | [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) | `models/t5_tokenizer/` |

**镜像站下载**（国内推荐）:
- HF-Mirror: `https://hf-mirror.com/<repo>/resolve/main/<filename>`
- 示例: `https://hf-mirror.com/Qwen/Qwen3-0.6B-Base/resolve/main/model.safetensors`

### 2. 准备数据集

支持两种标签格式：

**TXT 格式（传统）**:
```
dataset/
├── image001.jpg
├── image001.txt    # Danbooru 风格标签
└── ...
```

**JSON 格式（推荐）**:
```
dataset/
├── image001.jpg
├── image001.json   # 结构化标签
└── ...
```

JSON 标签示例：
```json
{
  "quality": "newest, safe",
  "count": "1girl",
  "character": "hatsune miku",
  "series": "vocaloid",
  "artist": "@wlop",
  "appearance": ["long hair", "blue hair", "twintails", "blue eyes"],
  "tags": ["singing", "microphone", "dynamic pose"],
  "environment": ["concert stage", "spotlight", "crowd"],
  "nl": "Miku performs energetically on stage."
}
```

JSON 格式支持**分类 shuffle**（appearance/tags/environment 各自内部打乱，固定字段保持在前），详见 [docs/json-caption-format.md](docs/json-caption-format.md)

### 3. 编辑配置文件

```bash
cp config/train_template.yaml config/my_training.yaml
# 编辑 my_training.yaml
```

### 4. 开始训练

```bash
python anima_train.py --config ./config/my_training.yaml
```

命令行参数可覆盖配置文件：

```bash
python anima_train.py --config ./config/my_training.yaml --lr 5e-5 --epochs 20
```

### 训练监控面板

- **默认地址**：`http://127.0.0.1:8765/`
- **关闭监控**：`--no-monitor`
- **不自动打开浏览器**：`--no-browser`
- **局域网/云端访问**：`--monitor-host 0.0.0.0`（并开放端口）

> 安全提醒：监控面板**没有鉴权**。不建议直接暴露到公网；云端建议用 SSH 端口转发访问。

## ⚙️ 配置说明

### 基础配置

```yaml
# 模型路径
transformer_path: "models/transformers/anima-preview.safetensors"
vae_path: "models/vae/qwen_image_vae.safetensors"
text_encoder_path: "models/text_encoders"
t5_tokenizer_path: "models/t5_tokenizer"

# 数据集
data_dir: "./dataset"
resolution: 1024
repeats: 10              # 数据重复次数

# Caption 处理
shuffle_caption: true    # 打乱标签顺序
keep_tokens: 0           # 保护前 N 个标签不打乱
prefer_json: false       # 优先使用 JSON 标签文件
flip_augment: false      # 水平翻转增强
tag_dropout: 0.0         # 标签随机丢弃概率
cache_latents: true      # 缓存 VAE latent
```

### LoRA/LoKr 配置

```yaml
lora_type: "lokr"        # lora 或 lokr
lora_rank: 32            # LoRA rank（建议 16-64）
lora_alpha: 32           # 通常与 rank 相同
lokr_factor: 8           # LoKr 专用参数
```

**选择建议**:
| 场景 | 类型 | Rank | 说明 |
|------|------|------|------|
| 单角色 LoRA | lora | 16-32 | 参数少，泛化好 |
| 画风 LoRA | lora | 8-16 | 低 rank 防止过拟合 |
| 多角色/复杂画风 | lokr | 32-64 | LyCORIS 更强表达力 |

### 训练参数

```yaml
epochs: 50
max_steps: 0             # 0 = 不限制
batch_size: 1
grad_accum: 4            # 有效 batch = batch_size × grad_accum
learning_rate: 1e-4
mixed_precision: "bf16"  # fp16, bf16, 或 no
grad_checkpoint: true    # 梯度检查点（省显存）
xformers: false          # Windows 5090 用 SDPA 更好
num_workers: 0           # Windows 必须为 0
```

### 保存与断点续训

```yaml
output_dir: "./output"
output_name: "my_lora"

# === 保存配置（重要！） ===
save_every: 0              # 每 N epoch 保存 LoRA (0=禁用)
save_every_steps: 500      # 每 N step 保存 LoRA (推荐)
save_state_every: 1000     # 每 N step 保存完整训练状态（可断点续训）

# === 继续训练 ===
resume_lora: ""            # 从已有 LoRA 继续训练
resume_state: ""           # 从训练状态恢复（断点续训）

seed: 42
```

**保存文件说明**：
- `{name}_step{N}.safetensors` - LoRA 权重，可直接在 ComfyUI 使用
- `training_state_step{N}.pt` - 完整训练状态（优化器、随机数、loss 历史）

### 采样配置

```yaml
sample_every: 5          # 每 N 个 epoch 采样
sample_steps: 0          # 或每 N step 采样
sample_infer_steps: 25
sample_cfg_scale: 4.0
sample_sampler_name: "er_sde"
sample_scheduler: "simple"

# 多提示词轮换
sample_prompts:
  - "newest, safe, 1girl, ..."
  - "newest, safe, 1boy, ..."
```

## 🔄 继续训练与断点恢复

### 从已有 LoRA 继续训练

如果你有一个训练好的 LoRA，想在此基础上继续训练：

```yaml
# 在配置文件中指定
resume_lora: "./output/my_lora_step1000.safetensors"
```

或命令行：

```bash
python anima_train.py --config config.yaml --resume-lora ./output/my_lora_step1000.safetensors
```

**注意**：这只加载 LoRA 权重，优化器状态会重置，学习率从头开始。

### 从中断处完全恢复（断点续训）

如果训练中断，想从**完全相同的状态**恢复（包括优化器、随机数、loss 历史）：

```yaml
# 在配置文件中指定
resume_state: "./output/cosmic_kaguya/training_state_step1000.pt"
```

或命令行：

```bash
python anima_train.py --config config.yaml --resume-state ./output/training_state_step1000.pt
```

**恢复内容**：
- ✅ LoRA 权重
- ✅ 优化器状态（momentum、Adam state）
- ✅ 随机数状态（torch、numpy、python random）
- ✅ 当前 epoch 和 step
- ✅ Loss 历史

### Ctrl+C 安全中断

训练时按 `Ctrl+C` 会**自动保存**：
```
检测到 Ctrl+C，正在保存训练状态...
已保存！下次使用 --resume-state "xxx/training_state_step1234.pt" 继续训练
```

### 推荐配置

```yaml
# 长时间训练推荐配置
save_every_steps: 500      # 每 500 step 保存 LoRA（方便选择最佳版本）
save_state_every: 2000     # 每 2000 step 保存训练状态（断点恢复用）
```

## 📁 配置示例

| 文件 | 场景 | 说明 |
|------|------|------|
| `config/train_template.yaml` | 通用模板 | 带详细注释，推荐作为起点 |
| `config/train_local.yaml` | 本地离线训练 | 所有路径指向本地模型 |

## 📖 文档

- [打标指南](docs/tagging-guide.md) - Anima 标签格式和最佳实践
- [JSON Caption 格式](docs/json-caption-format.md) - 结构化标签规范
- [训练技巧](docs/training-tips.md) - 常见问题和优化建议

## 🔧 工具脚本

| 脚本 | 功能 |
|------|------|
| `download_tokenizers.py` | 下载 tokenizer 文件（支持镜像） |
| `validate_local_models.py` | 验证本地模型文件是否正确 |
| `train_monitor.py` | 训练监控 Web 界面（训练时自动启动） |
| `convert_lokr_for_comfyui.py` | 转换其他工具导出的 LoKr 为 ComfyUI 格式 |

### convert_lokr_for_comfyui.py

将 `lycoris_` 前缀的 LoKr 权重转换为 ComfyUI 兼容的 `lora_unet_` 前缀。

> **注意**：本工具训练的 LoRA 已经是 ComfyUI 格式，**无需转换**。此脚本仅用于转换其他工具（如 kohya）导出的旧格式。

```bash
# 转换单个文件
python convert_lokr_for_comfyui.py ./my_lokr.safetensors

# 指定输出路径
python convert_lokr_for_comfyui.py ./my_lokr.safetensors --output ./converted.safetensors
```

| 输入格式 | 输出格式 | 说明 |
|----------|----------|------|
| `lycoris_xxx.lokr_w1` | `lora_unet_xxx.lokr_w1` | 自动转换前缀 |
| `lora_unet_xxx` | `lora_unet_xxx` | 已是正确格式，保持不变 |

## 💻 硬件要求

- **GPU**: 24GB+ 显存 (RTX 3090/4090/5090)
- **RAM**: 32GB+
- **存储**: SSD 推荐（latent 缓存）

## 🙏 致谢

- [FHfanshu/Anima_Trainer](https://github.com/FHfanshu/Anima_Trainer) - 原版训练脚本，本项目的基础
- [CircleStone Labs](https://huggingface.co/circlestone-labs) - Anima 模型开发团队
- [Comfy Org](https://github.com/comfyanonymous/ComfyUI) - ComfyUI 框架

## 📄 License

本项目整体以 **GPL-3.0** 发布（包含/派生自 ComfyUI 的 GPL-3.0 代码实现）。

同时，本项目包含部分来自第三方的代码/实现片段（例如 NVIDIA Cosmos / Wan2.1 等），请保留其文件头声明，并参考：

- `LICENSE`（GPL-3.0）
- `LICENSE-APACHE`（Apache-2.0 文本，用于仓库内 Apache-2.0 组件）
- `THIRD_PARTY_NOTICES.md`

**注意**：模型权重（例如 Anima / Qwen / VAE）通常有各自的条款（含 Non-Commercial 等限制），请以对应模型卡/仓库协议为准。
