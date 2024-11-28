# dataset settings
dataset_type = 'TamperedTextDataset'
data_root = 'data/ttd/RealTextMan/'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='val.txt',
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='test.txt',
        pipeline=test_pipeline))

val_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore', 'aFscore'])
test_evaluator = dict(type='BinaryIoUMetric', iou_metrics=['mIoU', 'mFscore'])
