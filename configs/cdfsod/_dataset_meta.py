# =====================================================================
# CD-FSOD Benchmark - Dataset Metadata
# ---------------------------------------------------------------------
# Single source of truth for the 6 standard datasets used in
# CDFSOD-benchmark (Fu et al., ECCV 2024) and the NTIRE 2025 CD-FSOD
# challenge. Each entry specifies:
#   - name        : short identifier (matches benchmark naming)
#   - data_root   : path relative to the mmdetection workdir
#                   (training is launched from ETS/mmdetection/, so
#                    '../datasets/<NAME>/' resolves to ETS/datasets/<NAME>/)
#   - classes     : tuple of class names exactly matching the COCO
#                   'categories' field of the dataset's annotation files
#                   (verified against datasets/<NAME>/annotations/1_shot.json).
#                   The fine-tuned expert is queried with these names.
#   - pre_aliases : optional same-length tuple of *English common-noun*
#                   aliases used when the PRETRAINED GroundingDINO expert
#                   is queried during PoE inference. Pretrained GD was
#                   trained on Objects365 + GoldG and has effectively no
#                   alignment for Latin order names ('Coleoptera') or
#                   concatenated tokens ('baseballfield'); rephrasing
#                   them as 'beetle' / 'baseball field' restores a
#                   useful base prior. Set to ``None`` when the original
#                   classes are already common English nouns (clipart1k,
#                   fish), in which case PoE uses the same prompt for
#                   both experts.
# =====================================================================

DATASETS = {
    'artaxor': dict(
        name='ArTaxOr',
        data_root='../datasets/ArTaxOr/',
        classes=(
            'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera',
            'Hymenoptera', 'Lepidoptera', 'Odonata',
        ),
        # Latin insect orders -> English common names. GoldG (web text
        # captions) almost never contains 'Hymenoptera'; 'wasp' is
        # ubiquitous. Aliases follow Wikipedia's canonical English name
        # for each order.
        pre_aliases=(
            'spider', 'beetle', 'fly', 'true bug',
            'wasp', 'butterfly', 'dragonfly',
        ),
    ),
    'clipart1k': dict(
        name='clipart1k',
        data_root='../datasets/clipart1k/',
        classes=(
            'sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa',
            'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train',
            'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus',
            'tvmonitor', 'horse',
        ),
        # VOC classes are already plain English nouns -- no aliasing needed.
        pre_aliases=None,
    ),
    'dior': dict(
        name='DIOR',
        data_root='../datasets/DIOR/',
        classes=(
            'Expressway-Service-area', 'Expressway-toll-station',
            'airplane', 'airport', 'baseballfield', 'basketballcourt',
            'bridge', 'chimney', 'dam', 'golffield', 'groundtrackfield',
            'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
            'tenniscourt', 'trainstation', 'vehicle', 'windmill',
        ),
        # DIOR class names are concatenated / hyphenated. GroundingDINO is
        # word-piece tokenised so 'baseballfield' fragments oddly; the
        # space-separated common phrase is far more aligned with GoldG.
        pre_aliases=(
            'highway service area', 'highway toll station',
            'airplane', 'airport', 'baseball field', 'basketball court',
            'bridge', 'chimney', 'dam', 'golf field', 'track field',
            'harbor', 'overpass', 'ship', 'stadium', 'storage tank',
            'tennis court', 'train station', 'vehicle', 'windmill',
        ),
    ),
    'fish': dict(
        name='FISH',
        data_root='../datasets/FISH/',
        classes=('fish',),
        pre_aliases=None,
    ),
    'neu-det': dict(
        name='NEU-DET',
        data_root='../datasets/NEU-DET/',
        classes=(
            'crazing', 'inclusion', 'patches', 'pitted_surface',
            'rolled-in_scale', 'scratches',
        ),
        # Steel-surface defect technical terms -> common-English nouns.
        # 'crazing' / 'pitted_surface' / 'rolled-in_scale' are domain-
        # specific; 'crack' / 'pit' / 'scale' are ubiquitous in GoldG.
        pre_aliases=(
            'crack', 'inclusion', 'patch', 'pit',
            'scale', 'scratch',
        ),
    ),
    'uodd': dict(
        name='UODD',
        data_root='../datasets/UODD/',
        classes=('seacucumber', 'seaurchin', 'scallop'),
        # Concatenated marine-creature names -> space-separated phrases.
        pre_aliases=('sea cucumber', 'sea urchin', 'scallop'),
    ),
}


def get_pre_aliases(dataset: str):
    """Return the pretrained-expert prompt aliases for a dataset, or None.

    Used by ``tools/poe_inference.py`` to build a *separate* text prompt for
    the pretrained expert. Returns ``None`` when the dataset's `classes`
    are already plain English nouns (no alias needed).
    """
    meta = DATASETS.get(dataset)
    if meta is None:
        return None
    aliases = meta.get('pre_aliases')
    if aliases is None:
        return None
    classes = meta['classes']
    assert len(aliases) == len(classes), (
        f"pre_aliases for {dataset!r} has {len(aliases)} entries but "
        f"classes has {len(classes)}; they must be 1:1 by position."
    )
    return tuple(aliases)

SHOTS = (1, 5, 10)

# Order in which datasets/shots appear in summary tables (matches
# CDFSOD-benchmark/main_results.sh)
DATASET_ORDER = ['artaxor', 'dior', 'fish', 'clipart1k', 'neu-det', 'uodd']
