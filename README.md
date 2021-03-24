# Yolov3-Plant

病害检测数据集处理
Clone this Repo:
```
git clone https://github.com/Huihuihh/Yolov3-Plant.git
cd Yolov3-Plant
```

Preparing the Yolov3 plant Dataset:
```
python prepare_database.py
```

Change darknet dataset to COCO dataset:
修改对应的`phase`和`trainsetfile`值，获取train.json和valid.json
```
python darknet_to_coco.py
```
