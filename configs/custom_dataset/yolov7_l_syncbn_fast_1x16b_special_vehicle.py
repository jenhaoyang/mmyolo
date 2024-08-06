_base_ = '../yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'

max_epochs = 1000
data_root = 'data/special_vehicle/'
class_name = (
    'person',
    'bicycle',
    'car',
    'motorbike',
    'bus',
    'truck',
    'ambulance',
    'fire truck',
    'police',
    'trailer',
    'semi-trailer',
    'SUV',
    'VAN',
    'dump truck',
    'tractor',
    'tank truck',
)
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(20, 220, 60)]  # 画图时候的颜色，随便设置即可
)

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

# 根据自己的 GPU 情况，修改 batch size，YOLOv5-s 默认为 8卡 x 16bs
train_batch_size_per_gpu = 16
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4

save_epoch_intervals = 2  # 每 interval 轮迭代进行一次保存一次权重

#resume = True
#load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'  # noqa
load_from = '/home/ai_server/2005013/mmyolo/mmyolo/work_dirs/pretrain/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth'

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 16)

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls 会根据 num_classes 动态调整，但是 num_classes = 1 的时候，loss_cls 恒为 0
        # loss_cls=dict(loss_weight=0.5 *
        #               (num_classes / 80 * 3 / _base_.num_det_layers))
    ))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    pin_memory=True,  # 开启锁页内存，节省 CPU 内存拷贝时间
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file=data_root + 'annotations/val.json',
    metric='bbox')
# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=True,  # 只将模型输出转换为coco的 JSON 格式并保存
    outfile_prefix='./work_dirs/coco_detection/test')  # 要保存的 JSON 文件的前缀

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

visualizer = dict(
    dict(vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ]))  # noqa
