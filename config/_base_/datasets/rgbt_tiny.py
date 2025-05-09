# dataset settings
dataset_type = 'RGBTTinyDataset'
data_root = '/workspace/mmdetection/datasets/RGBT-Tiny/'

backend_args = None
mod_list = ["img", "ir_img"]
img_shape = (1333, 800)

train_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=mod_list, 
        # replace_str=['/00/', '/01/'],   # raw, aim
        backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiModalResize', scale=img_shape, keep_ratio=True, mod_list=mod_list),
    dict(type='MultiModalRandomFlip', prob=0.5, mod_list=mod_list),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction'))
]
test_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=mod_list, 
        backend_args=backend_args),
    dict(type='MultiModalResize', scale=img_shape, keep_ratio=True, mod_list=mod_list),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=dict(
            img='annotations_coco/instances_train2017.json',
            ir_img='annotations_coco/instances_01_train2017.json'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=dict(
            img='annotations_coco/instances_test2017.json',
            ir_img='annotations_coco/instances_01_test2017.json'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    # type='CocoMetric',
    # type='CocoSmallMetric',
    type='CocoSAFitMetric',
    ann_file=data_root + 'annotations_coco/instances_test2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator
