accumulative_counts = (2, )
auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
data_root = '/workspace/mmdetection/datasets/RGBT-Tiny/'
dataset_type = 'RGBTTinyDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=1000, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl', timeout=1800),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
extra_return = [
    'rgb',
    'ir',
]
img_shape = (
    1333,
    800,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
mod_list = [
    'img',
    'ir_img',
]
mode = 'loss'
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet',
        with_cp=False),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=7,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead_Mine'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=dict(img=[
            95.7,
            99.4,
            99.62,
        ], ir_img=[
            87.71,
            87.71,
            87.71,
        ]),
        pad_size_divisor=1,
        std=dict(img=[
            42.79,
            41.22,
            43.68,
        ], ir_img=[
            49.66,
            49.66,
            49.66,
        ]),
        type='MultiModalDetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        mode='loss',
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        extra_return=[
            'rgb',
            'ir',
        ],
        fusion_layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        ir_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        mode='loss',
        num_layers=6,
        rgb_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4))),
    extra_return=[
        'rgb',
        'ir',
    ],
    mode='loss',
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO_TransFusion',
    with_box_refine=True)
num_levels = 4
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=dict(
            img='annotations_coco/instances_test2017.json',
            ir_img='annotations_coco/instances_01_test2017.json'),
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/workspace/mmdetection/datasets/RGBT-Tiny/',
        pipeline=[
            dict(
                backend_args=None,
                mod_list=[
                    'img',
                    'ir_img',
                ],
                mod_path_mapping_dict=dict({
                    'RGBT-Tiny':
                    dict(
                        img=dict(org_key='images', target_key='images'),
                        ir_img=dict(org_key='images', target_key='images'))
                }),
                type='LoadMultiModalImages'),
            dict(
                is_fixscale=True,
                keep_ratio=True,
                mod_list=[
                    'img',
                    'ir_img',
                ],
                scale=(
                    800,
                    1333,
                ),
                type='MultiModalResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                mod_list=[
                    'img',
                    'ir_img',
                ],
                type='PackMultiModalDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='RGBTTinyDataset'),
    drop_last=False,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/workspace/mmdetection/datasets/RGBT-Tiny/annotations_coco/instances_test2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoSAFitMetric')
test_pipeline = [
    dict(
        backend_args=None,
        mod_list=[
            'img',
            'ir_img',
        ],
        mod_path_mapping_dict=dict({
            'RGBT-Tiny':
            dict(
                img=dict(org_key='images', target_key='images'),
                ir_img=dict(org_key='images', target_key='images'))
        }),
        type='LoadMultiModalImages'),
    dict(
        is_fixscale=True,
        keep_ratio=True,
        mod_list=[
            'img',
            'ir_img',
        ],
        scale=(
            800,
            1333,
        ),
        type='MultiModalResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        mod_list=[
            'img',
            'ir_img',
        ],
        type='PackMultiModalDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file=dict(
            img='annotations_coco/instances_train2017.json',
            ir_img='annotations_coco/instances_01_train2017.json'),
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/workspace/mmdetection/datasets/RGBT-Tiny/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                backend_args=None,
                mod_list=[
                    'img',
                    'ir_img',
                ],
                mod_path_mapping_dict=dict({
                    'RGBT-Tiny':
                    dict(
                        img=dict(org_key='images', target_key='images'),
                        ir_img=dict(org_key='images', target_key='images'))
                }),
                type='LoadMultiModalImages'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                mod_list=[
                    'img',
                    'ir_img',
                ],
                prob=0.5,
                type='MultiModalRandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            mod_list=[
                                'img',
                                'ir_img',
                            ],
                            resize_type='MultiModalResize',
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            mod_list=[
                                'img',
                                'ir_img',
                            ],
                            resize_type='MultiModalResize',
                            scales=[
                                (
                                    400,
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            mod_list=[
                                'img',
                                'ir_img',
                            ],
                            type='MultiModalRandomCrop'),
                        dict(
                            keep_ratio=True,
                            mod_list=[
                                'img',
                                'ir_img',
                            ],
                            resize_type='MultiModalResize',
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                ),
                mod_list=[
                    'img',
                    'ir_img',
                ],
                type='PackMultiModalDetInputs'),
        ],
        return_classes=True,
        type='RGBTTinyDataset'),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        mod_list=[
            'img',
            'ir_img',
        ],
        mod_path_mapping_dict=dict({
            'RGBT-Tiny':
            dict(
                img=dict(org_key='images', target_key='images'),
                ir_img=dict(org_key='images', target_key='images'))
        }),
        type='LoadMultiModalImages'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(mod_list=[
        'img',
        'ir_img',
    ], prob=0.5, type='MultiModalRandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    mod_list=[
                        'img',
                        'ir_img',
                    ],
                    resize_type='MultiModalResize',
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    mod_list=[
                        'img',
                        'ir_img',
                    ],
                    resize_type='MultiModalResize',
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    mod_list=[
                        'img',
                        'ir_img',
                    ],
                    type='MultiModalRandomCrop'),
                dict(
                    keep_ratio=True,
                    mod_list=[
                        'img',
                        'ir_img',
                    ],
                    resize_type='MultiModalResize',
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
        ),
        mod_list=[
            'img',
            'ir_img',
        ],
        type='PackMultiModalDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=dict(
            img='annotations_coco/instances_test2017.json',
            ir_img='annotations_coco/instances_01_test2017.json'),
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/workspace/mmdetection/datasets/RGBT-Tiny/',
        pipeline=[
            dict(
                backend_args=None,
                mod_list=[
                    'img',
                    'ir_img',
                ],
                mod_path_mapping_dict=dict({
                    'RGBT-Tiny':
                    dict(
                        img=dict(org_key='images', target_key='images'),
                        ir_img=dict(org_key='images', target_key='images'))
                }),
                type='LoadMultiModalImages'),
            dict(
                is_fixscale=True,
                keep_ratio=True,
                mod_list=[
                    'img',
                    'ir_img',
                ],
                scale=(
                    800,
                    1333,
                ),
                type='MultiModalResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                mod_list=[
                    'img',
                    'ir_img',
                ],
                type='PackMultiModalDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='RGBTTinyDataset'),
    drop_last=False,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/workspace/mmdetection/datasets/RGBT-Tiny/annotations_coco/instances_test2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoSAFitMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'results/safit/rgbt_tiny-fusion/12e/dino_transfusion/back-add_enco-Fv6Fi6_v6i6f6_HfHvHi_deco-f6v6i6/'
