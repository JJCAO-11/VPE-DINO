# Copyright (c) OpenMMLab. All rights reserved.
"""Safe wrapper around mmengine.EMAHook that gracefully handles
checkpoints which lack the ``state_dict`` wrapper key.

Background
----------
``mmengine.hooks.EMAHook.after_load_checkpoint`` (mmengine 0.10.x)
unconditionally does ``copy.deepcopy(checkpoint['state_dict'])`` to
initialise its shadow EMA copy.  However, several mmdet-converted
pretrained checkpoints -- notably the GroundingDINO weights at
``groundingdino_swinb_cogcoor_mmdet-55949c9c.pth`` -- are stored as a
flat state-dict (the top-level keys ARE the parameter names directly,
no ``state_dict`` wrapper).  ``load_checkpoint_to_model`` already
handles this with ``checkpoint.get('state_dict', checkpoint)``, but the
EMAHook does NOT, so training crashes immediately with
``KeyError: 'state_dict'`` when ``load_from`` points at such a file.

This subclass wraps the raw flat checkpoint into ``{'state_dict': ...}``
before delegating to the parent implementation, fixing the crash
without changing EMA behaviour in any other way.
"""
from mmengine.hooks import EMAHook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class EMAHookSafe(EMAHook):
    """``EMAHook`` variant that tolerates flat (un-wrapped) checkpoints.

    Behaves identically to :class:`mmengine.hooks.EMAHook` except that
    if the loaded checkpoint dict has no ``'state_dict'`` key (i.e. its
    top-level keys ARE the model parameter names), we transparently
    wrap it before initialising the EMA shadow copy.
    """

    def after_load_checkpoint(self, runner, checkpoint):  # type: ignore[override]
        if 'state_dict' not in checkpoint:
            # Wrap raw flat state-dict so the parent class can find it.
            # Note: we deliberately build a NEW dict here rather than
            # mutating ``checkpoint`` in-place to avoid side-effects
            # downstream (e.g. other hooks that read the original
            # checkpoint structure).
            checkpoint = {'state_dict': checkpoint}
        super().after_load_checkpoint(runner, checkpoint)
