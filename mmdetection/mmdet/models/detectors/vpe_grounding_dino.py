"""VPE-DINO: Visual-Prompt-Enhanced GroundingDINO for CD-FSOD.

Inherits from :class:`GroundingDINO` and injects K-shot visual prompt
tokens into the language stream right after the text-feature projection.
The downstream encoder, decoder, and contrastive classification head are
unchanged: the visual prompts simply become extra "tokens" that the
contrastive head can dot-product with each detection query.

Architecture (see ``docs/vpe_dino_design.md`` for the full derivation):

    text_dict['embedded']          : [B, L, D]
    + visual prompts v_c (per class) : [B, C, D]
    -> concat                       : [B, L+C, D]

Each class c also adds position ``L+c`` to its entry in
``positive_maps`` / ``token_positive_maps`` so the loss / inference
correctly attribute the new logit slot to class c.

Pre-computed support features (``[C, K_max, D_swin]``) are loaded from
``support_features_path`` (built by ``tools/build_support_features.py``).
The only new trainable parameter is ``vpe_proj`` (Linear(D_swin, D_text)).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from .grounding_dino import GroundingDINO


@MODELS.register_module()
class VPEGroundingDINO(GroundingDINO):
    """GroundingDINO + K-shot visual prompt tokens.

    Args:
        support_features_path (str | None): Path to the ``.pt`` bundle
            produced by ``tools/build_support_features.py``. Loaded as a
            non-trainable buffer. When ``None`` the model behaves exactly
            like the parent ``GroundingDINO`` (useful for ablation).
        vpe_dim (int | None): Output dim of ``vpe_proj``. Defaults to the
            text-feature dim inferred from ``text_feat_map.out_features``
            (256 for the standard config).
        vpe_pool (str): How to collapse the K-shot dimension of the
            support features into a single per-class vector. One of
            ``mean`` (default) or ``max``.
        vpe_disable (bool): If ``True``, skip injection at runtime even
            when ``support_features`` is loaded. Convenient for sanity
            checks via ``--cfg-options model.vpe_disable=True`` in
            ``tools/test.py``.
    """

    def __init__(self,
                 *args,
                 support_features_path: Optional[str] = None,
                 vpe_dim: Optional[int] = None,
                 vpe_pool: str = 'mean',
                 vpe_disable: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vpe_disable = vpe_disable
        self.vpe_pool = vpe_pool

        # Resolve text feature dim ----------------------------------------
        if hasattr(self, 'text_feat_map') and self.text_feat_map is not None:
            text_dim = self.text_feat_map.out_features
        else:
            text_dim = 256
        self._text_dim = text_dim

        # Load support features as non-trainable buffer -------------------
        if support_features_path:
            sf_path = Path(support_features_path)
            assert sf_path.exists(), (
                f'support_features_path does not exist: {sf_path}')
            bundle = torch.load(sf_path, map_location='cpu')
            features = bundle['features']           # [C, K_max, D_swin]
            assert features.dim() == 3, (
                f'expected [C, K, D], got {tuple(features.shape)}')
            self.register_buffer('support_features', features.contiguous())
            self._vpe_classes: List[str] = list(bundle.get('classes', []))
            self._vpe_K_per_class: List[int] = list(
                bundle.get('K_per_class', [features.shape[1]] * features.shape[0]))
            d_swin = features.shape[-1]
            print(f'[vpe] loaded support features {tuple(features.shape)} '
                  f'(classes={self._vpe_classes}) from {sf_path}')
        else:
            self.register_buffer('support_features', torch.empty(0))
            self._vpe_classes = []
            self._vpe_K_per_class = []
            d_swin = 1024  # placeholder; vpe_proj initialised but unused

        # The only new trainable layer ------------------------------------
        out_dim = vpe_dim or text_dim
        self.vpe_proj = nn.Linear(d_swin, out_dim, bias=True)
        # Zero-init so initial visual prompts are no-ops; gradient is then
        # free to grow them. Stabilises the first few iterations.
        nn.init.zeros_(self.vpe_proj.weight)
        nn.init.zeros_(self.vpe_proj.bias)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _vpe_active(self) -> bool:
        """Return True iff visual prompts should be injected this call."""
        if self.vpe_disable:
            return False
        if not isinstance(self.support_features, Tensor):
            return False
        return self.support_features.numel() > 0

    def _build_visual_prompts(self, batch_size: int,
                              device: torch.device) -> Tensor:
        """Pool support features over K then project to text dim.

        Returns:
            Tensor of shape ``[B, C, D_text]``.
        """
        sf = self.support_features.to(device)        # [C, K_max, D_swin]
        if self.vpe_pool == 'mean':
            # K_per_class can vary; mask padding rows by their actual K.
            v_per_class = []
            for c, k in enumerate(self._vpe_K_per_class):
                k = max(1, int(k))
                v_per_class.append(sf[c, :k].mean(dim=0))
            v = torch.stack(v_per_class, dim=0)        # [C, D_swin]
        elif self.vpe_pool == 'max':
            v_per_class = []
            for c, k in enumerate(self._vpe_K_per_class):
                k = max(1, int(k))
                v_per_class.append(sf[c, :k].max(dim=0).values)
            v = torch.stack(v_per_class, dim=0)        # [C, D_swin]
        else:
            raise ValueError(f'unknown vpe_pool: {self.vpe_pool!r}')

        v = self.vpe_proj(v)                            # [C, D_text]
        return v.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _inject_into_text_dict(self, text_dict: Dict[str, Tensor],
                               v: Tensor) -> Tuple[Dict[str, Tensor], int]:
        """Append C visual-prompt tokens to ``text_dict`` and extend masks.

        Args:
            text_dict: Output of language_model + text_feat_map. Mutated
                in place; returned for clarity.
            v: ``[B, C, D]`` visual prompts.

        Returns:
            (mutated text_dict, L_text_pre_injection). The caller uses
            ``L_text`` to know where to add visual-prompt token positions
            inside ``positive_maps`` / ``token_positive_maps``.
        """
        emb = text_dict['embedded']                     # [B, L, D]
        B, L, D = emb.shape
        C = v.shape[1]
        assert v.shape[0] == B and v.shape[2] == D, (
            f'visual prompt shape mismatch: {tuple(v.shape)} vs text '
            f'embedded {tuple(emb.shape)}')

        # Concat along sequence dim ---------------------------------------
        text_dict['embedded'] = torch.cat([emb, v], dim=1)

        # text_token_mask: append C ones (visual prompts are always valid).
        if 'text_token_mask' in text_dict:
            mask = text_dict['text_token_mask']         # [B, L] bool / int
            ones = mask.new_ones((B, C))
            text_dict['text_token_mask'] = torch.cat([mask, ones], dim=1)

        # position_ids: continue numbering from L. Some configs have this,
        # some don't. We only extend if present.
        if 'position_ids' in text_dict:
            pos = text_dict['position_ids']             # [B, L]
            extra = (torch.arange(L, L + C, device=pos.device, dtype=pos.dtype)
                     .unsqueeze(0).expand(B, -1))
            text_dict['position_ids'] = torch.cat([pos, extra], dim=1)

        # text self-attention mask: visual prompts are mutually visible
        # AND visible to all text tokens (simplifies bookkeeping; the
        # original mask only restricts text-to-text attention to keep
        # category chunks isolated, which is irrelevant for the visual
        # tokens we just appended).
        if 'masks' in text_dict and text_dict['masks'] is not None:
            m = text_dict['masks']                       # [B, L, L] bool
            new_total = L + C
            # Build [B, L+C, L+C] starting from the original mask block.
            new_mask = m.new_ones((B, new_total, new_total))
            new_mask[:, :L, :L] = m
            # Original text tokens see new visual tokens too (set True).
            new_mask[:, :L, L:] = True
            new_mask[:, L:, :] = True
            text_dict['masks'] = new_mask

        return text_dict, L

    @staticmethod
    def _augment_positive_maps(positive_maps: List[Tensor],
                               gt_labels: List[Tensor],
                               L_text: int) -> List[Tensor]:
        """Add visual-prompt token positions to per-image ``positive_maps``.

        ``positive_maps[i]`` has shape ``[num_gt_in_image, max_text_len]``.
        For every gt of class ``c`` we set position ``L_text + c`` to 1
        (then re-normalise the row so each row still sums to 1 over its
        active tokens).
        """
        out = []
        for pm, labels in zip(positive_maps, gt_labels):
            pm = pm.clone()                              # [num_gt, max_text_len]
            for j, lbl in enumerate(labels.tolist()):
                pos = L_text + int(lbl)
                if 0 <= pos < pm.shape[1]:
                    pm[j, pos] = 1.0
            # Re-normalise each row so it remains a token-distribution.
            row_sum = pm.sum(dim=-1, keepdim=True).clamp_min_(1e-6)
            pm = pm / row_sum
            out.append(pm)
        return out

    @staticmethod
    def _augment_token_positive_maps(
        token_positive_maps,
        L_text: int,
    ):
        """At inference, each image carries a ``token_positive_maps`` dict
        mapping ``class_idx -> [token_position_list]``.  Append L_text+c
        to class c's list.
        """
        if token_positive_maps is None:
            return token_positive_maps
        out = copy.deepcopy(token_positive_maps)
        for c, positions in list(out.items()):
            # GD numbers labels starting from 1 in this dict (see
            # ``create_positive_map_label_to_token(..., plus=1)``); class
            # index 0 maps to label 1, etc.  We mirror that offset.
            class_idx_zero_based = int(c) - 1
            new_pos = L_text + class_idx_zero_based
            if isinstance(positions, list):
                if new_pos not in positions:
                    out[c] = positions + [new_pos]
        return out

    # ------------------------------------------------------------------
    # Override loss() to inject right after text_feat_map
    # ------------------------------------------------------------------
    def loss(self, batch_inputs: Tensor, batch_data_samples: OptSampleList):
        """Identical to ``GroundingDINO.loss`` except for the VPE
        injection step between ``text_feat_map`` and ``forward_transformer``.

        Code paths copied from
        ``mmdet/models/detectors/grounding_dino.py:GroundingDINO.loss``.
        """
        text_prompts = []
        for data_samples in batch_data_samples:
            text_tuple = data_samples.text
            text_prompts.append(text_tuple)

        gt_labels = [data_samples.gt_instances.labels
                     for data_samples in batch_data_samples]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        # ============= VPE INJECTION =====================================
        if self._vpe_active():
            B = text_dict['embedded'].shape[0]
            v = self._build_visual_prompts(B, text_dict['embedded'].device)
            text_dict, L_text = self._inject_into_text_dict(text_dict, v)
            positive_maps = self._augment_positive_maps(
                positive_maps, gt_labels, L_text)
        # =================================================================

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        head_inputs_dict = self.forward_transformer(
            visual_features, text_dict, batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    # ------------------------------------------------------------------
    # Override predict() to inject in BOTH non-chunked and chunked paths
    # ------------------------------------------------------------------
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        if len(text_prompts) == 1:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities,
                    enhanced_text_prompts[0], tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    tp, custom_entities, ep, tpos)
                for tp, ep, tpos in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)
        token_positive_maps = list(token_positive_maps)

        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked path -- bs=1 supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []
            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                # ============= VPE INJECTION (chunked) ===================
                if self._vpe_active():
                    Bi = text_dict['embedded'].shape[0]
                    v = self._build_visual_prompts(
                        Bi, text_dict['embedded'].device)
                    text_dict, L_text = self._inject_into_text_dict(
                        text_dict, v)
                    token_positive_maps_once = (
                        self._augment_token_positive_maps(
                            token_positive_maps_once, L_text))
                # =========================================================

                batch_data_samples[0].token_positive_map = \
                    token_positive_maps_once
                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # non-chunked path
            text_dict = self.language_model(list(text_prompts))
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            # ============= VPE INJECTION (non-chunked) ===================
            if self._vpe_active():
                B = text_dict['embedded'].shape[0]
                v = self._build_visual_prompts(
                    B, text_dict['embedded'].device)
                text_dict, L_text = self._inject_into_text_dict(text_dict, v)
                token_positive_maps = [
                    self._augment_token_positive_maps(tpm, L_text)
                    for tpm in token_positive_maps
                ]
            # =============================================================

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # Wrap results into batch_data_samples (parent does this too;
        # mirror the same code rather than calling super so we keep a
        # single override surface).
        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
