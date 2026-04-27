_base_ = ['../_base_vpe_recover.py']

data_root = '../datasets/ArTaxOr/'
metainfo = dict(classes=('Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Lepidoptera', 'Odonata'))
train_ann_file = 'annotations/1_shot.json'
num_classes = 7

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=metainfo))

val_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=7), support_features_path='../support_features/artaxor_1shot.pt')

work_dir = 'work_dirs/cdfsod/vpe/artaxor_1shot_recover50_v2'
