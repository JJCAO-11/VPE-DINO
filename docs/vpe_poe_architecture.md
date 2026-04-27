# VPE-DINO + PoE 架构总览

本图描述当前 ETS 仓库中相对于原始 **GroundingDINO Swin-B** 的两处改进：

1. **训练侧 / VPE-DINO**：在 `text_feat_map` 之后，向语言流末尾注入 K-shot **视觉 prompt token**，让对比头同时利用文本相似度与视觉相似度。
2. **推理侧 / PoE**：用 fine-tuned 模型与原始预训练 GD（可选用英文俗名 alias prompt）做 score-fusion。

代码入口：

- 模型：`mmdetection/mmdet/models/detectors/vpe_grounding_dino.py`
- 离线 support：`tools/build_support_features.py`
- 推理融合：`tools/poe_inference.py` / `tools/poe_run_all.py`
- alias 词表：`configs/cdfsod/_dataset_meta.py`

---

## 1. 端到端总览

```mermaid
flowchart LR
    subgraph OFF["离线一次性预处理"]
        A1[K-shot 训练图片<br/>+ bbox]
        A2[Crop & Resize 224x224]
        A3[Swin-B backbone<br/>frozen, GD 预训练权重]
        A4[Global Avg Pool<br/>到 D_swin=1024]
        A5[(support_features<br/>.pt: C, K, 1024)]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph TRAIN["训练 / 推理前向"]
        B1[输入图像 H x W x 3]
        B2[Swin-B Image Backbone]
        B3[Channel Mapper Neck<br/>4 个 level]

        T1["类名文本 prompt<br/>'spider. beetle. ...'"]
        T2[BERT base-uncased]
        T3[text_feat_map<br/>Linear 768 to 256]
        T4["text_dict.embedded<br/>shape B, L, 256"]

        VPE["**VPE 注入**<br/>vpe_proj 1024 to 256<br/>+ concat C 个视觉 token"]
        T5["text_dict.embedded<br/>shape B, L+C, 256"]

        FUS[Cross-modal Fusion Encoder<br/>Image x Text+Visual]
        DEC[Transformer Decoder<br/>900 queries x memory]
        HEAD[ContrastiveEmbed Head<br/>query DOT memory_text]
        OUT[每 query 对每 token 的 logit<br/>positive_maps 把 L+c 归到类 c]
    end

    A5 --> VPE
    B1 --> B2 --> B3 --> FUS
    T1 --> T2 --> T3 --> T4 --> VPE --> T5 --> FUS
    FUS --> DEC --> HEAD --> OUT
```

---

## 2. VPE 注入细节（shape 与流程）

```mermaid
flowchart TB
    S[("support_features<br/>C, K, 1024")]
    P1["按类 mean / max over K<br/>得到 v: C, 1024"]
    P2["vpe_proj Linear<br/>zero-init<br/>1024 -> 256"]
    P3["v_text: C, 256<br/>broadcast 到 B, C, 256"]
    S --> P1 --> P2 --> P3

    E0["text_dict.embedded<br/>B, L, 256"]
    M0["text_token_mask<br/>B, L"]
    PM0["positive_maps gt_i<br/>num_gt, max_text_len=256<br/>类 c gt 在文本 c-token 处=1"]

    E1["concat dim=1<br/>B, L+C, 256"]
    M1["concat 全 1<br/>B, L+C"]
    PM1["类 c gt 在位置 L+c 也置 1<br/>再 row-normalize"]

    E0 --> E1
    M0 --> M1
    PM0 --> PM1
    P3 --> E1
    P3 -. 决定 C .-> M1
    P3 -. 决定 C .-> PM1
```

> 约束：必须 `L + C <= 256` (`max_text_len`)。当前 6 个数据集最大 DIOR `C=20, L≈100`，远低于 256。

---

## 3. 训练前向中的注入位置（代码对照）

```text
GroundingDINO.loss(...)                                  # 父类
  ├─ language_model(text_prompts)                       # BERT
  ├─ text_feat_map(...)                                  # Linear 768 -> 256
  │
  │   <<<<<<<<<<<<<  VPE-DINO 注入点  >>>>>>>>>>>>>>>
  │   if support_features 已加载:
  │     v = vpe_proj(mean_K(support_features))   # [B, C, 256]
  │     text_dict['embedded'] = cat([emb, v])    # [B, L+C, 256]
  │     text_token_mask、position_ids、masks 同步扩展
  │     positive_maps[c-th gt][L+c] = 1, row-normalize
  │
  ├─ extract_feat(image)                                # Swin-B + Neck
  ├─ forward_transformer(visual, text_dict, ...)        # 不改
  └─ bbox_head.loss(...)                                # 不改
```

唯一新增可训练参数：`vpe_proj = nn.Linear(1024, 256)`，~262K，**zero-init** 保证训练第 0 步退化为原始 GD。

---

## 4. PoE 推理融合（可选）

```mermaid
flowchart LR
    IMG[(测试图片)]

    subgraph FT["Fine-tuned VPE-DINO"]
        FT1[VPEGroundingDINO<br/>load best ckpt]
        FT2["类名 prompt<br/>'Coleoptera. Diptera. ...'"]
        FT3[detections_ft:<br/>每个 box 含 score s_ft、label]
        FT1 --> FT3
        FT2 --> FT1
    end

    subgraph PRE["原始预训练 GD<br/>(no fine-tune)"]
        PRE1[GroundingDINO Swin-B<br/>cogcoor mmdet ckpt]
        PRE2["alias prompt<br/>'beetle. fly. ...'<br/>(同位置对齐)"]
        PRE3[detections_pre:<br/>每个 box 含 score s_pre]
        PRE1 --> PRE3
        PRE2 --> PRE1
    end

    IMG --> FT1
    IMG --> PRE1

    MATCH["IoU 配对 (label-agnostic)<br/>每个 ft box 取最强匹配的 pre box"]
    FUSE["log_linear PoE:<br/>log s_fused = (1-α) log s_ft + α log s_pre"]
    OUT2[(融合后 detections<br/>导出 COCO json)]

    FT3 --> MATCH
    PRE3 --> MATCH
    MATCH --> FUSE --> OUT2
```

支持的 fusion 模式：

| `--mode` | 公式 | 备注 |
|---|---|---|
| `log_linear` | `s_ft^(1-α) * s_pre^α` | 默认 / 严格 PoE |
| `multiplicative` | `s_ft * (1 + α * s_pre)` | 较温和 |
| `additive` | `(1-α) s_ft + α s_pre` | 不是 PoE，是 MoE |
| `gated` | `s_ft + α * 1[s_pre>τ] * s_pre` | 阈值开关 |

---

## 5. 数据流（一行总结）

```text
[K-shot crops] --Swin--> [C, K, 1024] --mean+vpe_proj--> [C, 256]
                                                          |
[image]   --Swin/Neck--> [4-level feats]                  |
                            \                             |
                             \---> Cross-modal Fusion <---/
                                          |
[text]    --BERT/Map--> [L, 256] -- concat ----------> [L+C, 256]
                                          |
                                       Decoder (900 queries)
                                          |
                                ContrastiveEmbed Head
                                          |
                          per-class score = max_token_in_class(query·memory_text)
```

---

## 6. 改进点与原始 GD 的差异

| 模块 | 原始 GroundingDINO | 当前改进版 |
|---|---|---|
| Image Backbone | Swin-B | 同上（不变） |
| Text Encoder | BERT-base | 同上（不变） |
| 文本投影 | `text_feat_map` Linear 768→256 | 同上 |
| **文本 token 数** | L（来自 prompt 分词） | **L + C**（追加 C 个视觉 prompt） |
| **新增可训练参数** | 0 | `vpe_proj` Linear(1024,256) ≈ 262K |
| Cross-modal Fusion / Decoder / Head | 不变 | 不变 |
| `positive_maps` | 仅文本 token 位置 | **额外** 在位置 `L+c` 置 1 |
| 推理融合 | 单模型 | **可选** PoE：fine-tuned ⊗ 预训练（alias prompt） |

---

## 7. 关键设计选择小抄

- **Backbone 复用**：visual prompt 用 GD 自带 Swin-B 提特征，零额外 backbone 参数，特征空间天然对齐。
- **离线预提取**：support feature 一次性算好存 `.pt`，训练循环里只走 `vpe_proj`，速度几乎与 baseline 持平。
- **Zero-init 投影**：`vpe_proj` 初始化为 0，第 0 步 `cls_logits` 与原 GD 完全一致，避免冷启动崩。
- **mean over K**：K 小时方差大，attention pool 收益有限；mean 简单且稳。
- **mask 全互通**：visual prompt 不参与 GD 原本的 category-chunk 隔离，简化 bookkeeping。
- **PoE alias prompt**：解决 GD 预训练词表（Objects365 + GoldG）对拉丁学名 / 拼接词的支持空洞，按位置对齐 ft / pre 的类 ID。
