# Third-Party Notices

本仓库包含/改写/派生了部分第三方代码与实现片段。请在分发时遵守其许可并保留必要的版权与许可声明。

## ComfyUI (GPL-3.0)

- **来源**：`comfyanonymous/ComfyUI`（现由 Comfy-Org 维护）
- **许可**：GPL-3.0
- **涉及文件**：
  - `models/anima_modeling.py`（实现结构与 ComfyUI 的 `comfy/ldm/anima/model.py` 高度相关）

> 由于包含/派生自 GPL-3.0 代码，本项目整体以 GPL-3.0 发布（见 `LICENSE`）。

## NVIDIA Cosmos (Apache-2.0)

- **来源**：NVIDIA 相关实现（文件内含 SPDX 头）
- **许可**：Apache-2.0（见文件头 `SPDX-License-Identifier: Apache-2.0`）
- **涉及文件**：
  - `models/cosmos_predict2_modeling.py`
  - `models/anima_modeling_core.py`

本仓库额外提供 `LICENSE-APACHE` 以便分发 Apache-2.0 许可文本。

## Alibaba Wan2.1 VAE（请再次确认上游许可）

- **来源**：`Wan-Video/Wan2.1` 的 VAE 实现（与 `wan/modules/vae.py` 对应）
- **涉及文件**：
  - `models/wan/vae2_1.py`

该文件头目前仅包含版权声明（未显式 SPDX）。上游仓库通常宣称 Apache-2.0，但建议你在开源前**再次核对上游仓库的 LICENSE/NOTICE**，确保分发合规。

---

如你希望把项目改为更宽松的许可（例如 MIT），需要先移除/替换所有 GPL-3.0 派生部分，并重新梳理第三方依赖的许可兼容性。

