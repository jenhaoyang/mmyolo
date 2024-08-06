- inference 指令
  configs/custom_dataset/yolov4_l_mish_1xb16-300e_5car.py  weights/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth

configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py  weights/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth

- 命名規則
  https://mmyolo.readthedocs.io/zh-cn/latest/tutorials/config.html?highlight=syncbn#id13

- Hook說明
  https://mmengine.readthedocs.io/zh-cn/latest/design/hook.html

```python
pre_hooks = [(print, 'hello')]
post_hooks = [(print, 'goodbye')]

def main():
    for func, arg in pre_hooks:
        func(arg)
    print('do something here')
    for func, arg in post_hooks:
        func(arg)

main()
```

預設鉤子說明
https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html

- 配置文件繼承說明
  https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#id3

`optimizer_cfg.py`

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

`runtime_cfg.py`

```python
gpu_ids = [0, 1]
```

對於字典類型的繼承修改
`resnet50.py`

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(lr=0.01)
```

`{'type': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}`

# batch_shapes_cfg

BatchShapePolicy 原始碼在 mmyolo/mmyolo/datasets/utils.py

https://mmyolo.readthedocs.io/zh-cn/latest/tutorials/config.html
