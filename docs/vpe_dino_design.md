# VPE-DINO 架构设计

## 1. 核心思想

VPE-DINO（Visual-Prompt-Enhanced GroundingDINO）= 在 GroundingDINO 的开放词表分类头上**追加视觉支持集（K-shot）信号**。

GroundingDINO 用一个 **对比分类头** `ContrastiveEmbed`：
```
cls_logits[b, q, k] = visual_query[b, q] @ text_feat[b, k]
```
然后用 `positive_maps[gt_id, k]`（指示 token-k 属于哪个类）把 token 级 logit 转成类别级分数。

VPE-DINO 不改这个数学结构，而是**给每个类 c 追加一个"视觉 token"** $v_c \in \mathbb{R}^{D_\text{text}}$ 到 `text_feat` 末尾：

```
text_feat: [B, L,    D]   →  [B, L+C, D]      (C = num_classes)
positive_maps[gt_with_class_c]: 在位置 L+c 也置 1
```

对比头自动学会用「文本相似度 + 视觉相似度」共同判定每个类，无需改动 encoder/decoder/head。

## 2. 数据流（精确到 mmdet 代码行）

`@mmdetection/mmdet/models/detectors/grounding_dino.py:559-562`：

```python
text_dict = self.language_model(new_text_prompts)            # BERT
if self.text_feat_map is not None:
    text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
# >>>>>>>>>>>>>> VPE 注入点 <<<<<<<<<<<<<<<<
# 在这里 augment text_dict + positive_maps
```

之后的 `forward_transformer` 用 `text_dict['embedded']` 做交叉注意力，head 用同样的 `memory_text` 做对比分类。**只要保证下游 shape 一致，无需改动它们**。

## 3. 视觉 prompt $v_c$ 的来源

```
对每个类 c, 取 K-shot 训练集里所有标注框 (image_k, bbox_k):
    crop_k = image_k[bbox_k]        (RGB, resize 到 224×224)
    feat_k = swin_b(crop_k)         (用 GD 自身的 Swin-B backbone, frozen)
    feat_k = global_avg_pool(feat_k)  (D_swin=1024)
v_c[support] = mean_k(feat_k)        (D_swin=1024)
v_c[text]    = vpe_proj(v_c[support])  (D_swin=1024 -> D_text=256)
```

**关键设计选择**：
| 选项 | 决策 | 理由 |
|---|---|---|
| Backbone | 用 GD 自带 Swin-B (frozen) | 共享特征空间，0 额外参数 |
| 池化 | mean over K shots | 简单，K 小时方差大，attention pool 收益有限 |
| 投影 | 1 层 Linear (1024→256) | 唯一新增可训练参数 (~262K) |
| 训练 | 投影层 + 现有 GD 全部参数 | 端到端微调 |
| 离线/在线 | **离线** 一次性提取 v_c[support] 缓存 | 训练加速 ×10+ |

## 4. mask & positive_map 处理

设 `L = text_dict['embedded'].shape[1]`（batch padded 文本长度），`C = num_classes`，注入后：

| 字段 | 原 shape | 新 shape | 新增内容 |
|---|---|---|---|
| `embedded` | [B, L, D] | [B, L+C, D] | 末尾 C 个视觉 token |
| `text_token_mask` | [B, L] | [B, L+C] | 末尾 C 个 1 |
| `position_ids` | [B, L] | [B, L+C] | L..L+C-1 (continue) |
| `masks` (text self-attn) | [B, L, L] | [B, L+C, L+C] | 视觉 token 之间 + 与所有文本 token 全互通 |
| `positive_maps[i]` | [num_gt, max_text_len=256] | [num_gt, max_text_len=256] | 类 c gt 在位置 L+c 加 1（重新 row-normalize） |

> **约束**：必须保证 `L + C ≤ max_text_len = 256`。当前 6 个数据集中：
> - DIOR (C=20, L≈100): 总 120 < 256 ✓
> - 其他 (C≤7, L≤50): 总 ≤ 57 < 256 ✓

## 5. 文件结构

```
ETS/
├── tools/
│   └── build_support_features.py        # 离线提取 v_c[support] -> *.pt
├── mmdetection/mmdet/models/detectors/
│   └── vpe_grounding_dino.py            # VPEGroundingDINO 继承 GroundingDINO
├── configs/cdfsod/
│   ├── _base_vpe.py                     # 第三个 base config
│   └── <ds>/<shot>shot_vpe.py           # 18 个新 leaf 配置
├── tools/
│   └── gen_cdfsod_configs.py            # 修改: 增加 vpe variant
└── support_features/                    # 离线缓存
    └── <ds>_<shot>shot.pt               # {features: [C,K,D], classes: List}
```

## 6. 训练设置

- **优化器、学习率**：与 baseline 完全相同（公平对比）
- **可训练参数**：原 GD 所有参数 + 新增 `vpe_proj` (1 层 Linear)
- **数据增强**：和 baseline 一致（无 ETS 增强，最公平）
- **训练 epochs**：50（同其他 variant）
- **Support 特征**：训练前一次性提取，不参与梯度

## 7. Smoke test 计划

1. 离线提取 `artaxor_1shot` 的支持集特征 → `support_features/artaxor_1shot.pt`
2. 验证 shape: `[7, 1, 1024]` (7 类 × 1 shot × Swin-B last stage dim)
3. 加载到 `VPEGroundingDINO`，跑 1 张图 forward → 检查 logits shape
4. 跑 1 个 batch backward → 检查 vpe_proj 收到梯度
5. 完整跑 1 个 epoch on artaxor 1shot → 不崩 + 训练 loss 下降

## 8. 实验矩阵

```
6 datasets × 3 shots × { baseline, ets, vpe } = 54 个实验
                                       ↑ 已有       ↑ 新增 18
```

VPE 训练时间 ≈ baseline（多 1 个 Linear 层、+ ~262K 参数 ≈ 0.5%）。

## 9. 风险 & 缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| `text_self_attention_masks` 修改不当导致训练崩溃 | 中 | 先用「全互通」简化 mask，再 ablate |
| 视觉/文本特征空间差异大，初期 vpe_proj 梯度爆炸 | 中 | 投影层 zero-init + warmup |
| K=1 时 v_c 噪声大，测试集分布偏移大 | 中 | 至少 K=5 时应明显有效 |
| 不收敛 / 反向降 mAP | 低 | smoke test 阶段就能发现 |

## 10. 进度追踪

见 `progress.txt` 或运行 `python tools/gen_cdfsod_configs.py --check vpe` 查询配置生成情况。
