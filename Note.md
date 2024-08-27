# 安裝套件注意事項

- 安裝albumentations可以試看看不要移除opencv，因為移除會讓opencv的RGB2BGR故障
- albumentations >= 1.14.11 要直接手動幫mmdetection寫補丁https://github.com/open-mmlab/mmdetection/pull/11870/files

# 常用指令

- 訓練指令
  python tools/train.py configs/custom_dataset/yolov4_l_mish_1xb16-300e_5car.py

- inference 指令
  configs/custom_dataset/yolov4_l_mish_1xb16-300e_5car.py  weights/last.pt

- 列印config
  python tools/misc/print_config.py configs/custom_dataset/yolov4_l_mish_1xb16-300e_5car.py --save-path /home/ai_server/2005013/mmyolo/mmyolo/my_exp/check_config.json

- 切割訓練集
  python tools/misc/coco_split.py --json data/eastcoast_5car/annotations/result.json \
  --out-dir ./data/debug_eastcoast_5car/annotations \
  --ratios 0.99 0.005 0.005\
  --shuffle

* 命名規則
  https://mmyolo.readthedocs.io/zh-cn/latest/tutorials/config.html?highlight=syncbn#id13

* Hook說明
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
