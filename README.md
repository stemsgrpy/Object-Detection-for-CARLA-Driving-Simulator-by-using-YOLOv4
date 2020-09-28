# Object-Detection-for-CARLA-Driving-Simulator-by-using-YOLOv4

## CARLA Simulator
- The simulation platform provides open digital assets (urban layouts, buildings, vehicles), as shown in Fig1.
- Download [CARLA](http://carla.org/) (CARLA_0.9.5 version)
- Running CARLA
```
./CarlaUE4.sh (Linux)
CarlaUE4.exe (Windows)
```

<p align="center">
  <img width="500" src="/README/carla.jpg">
</p>
<p align="center">
  Figure 1: Urban Layout
</p>

## Dataset
- CARLA Simulator contains different urban layouts and can also generate objects.
  - Urban layout **Town05** is used as experimental site
  - Objects (**Vehicle**, **Bike**, **Motobike**, **Traffic light**, **Traffic sign**) can be recognized in different urban layouts
- Download [Carla-Object-Detection-Dataset](https://github.com/DanielHfnr/Carla-Object-Detection-Dataset)
  - Put `.png` and `.xml` to the `VOCdevkit/VOC2007/JPEGImages` and `VOCdevkit/VOC2007/Annotations`, respectively
- Obtain label format: (2007_train.txt)
```
python voc_annotation.py
```

## Result
```
python predict.py
```
<p align="center">
  <img src="/README/Town03_013260_predict.png" alt="Description" width="160" height="95" border="0" />
  <img src="/README/Town03_015500_predict.png" alt="Description" width="160" height="95" border="0" />
  <img src="/README/Town04_002280_predict.png" alt="Description" width="160" height="95" border="0" />
  <img src="/README/Town05_017100_predict.png" alt="Description" width="160" height="95" border="0" />
</p>
<p align="center">
  Figure 2: Image Detection
</p>

```
python video.py
```
<p align="center">
  <a href="https://www.youtube.com/watch?v=Rf1Fd8reW7w" target="_blank">
    <img src="/README/video.gif" alt="Description" width="850" height="320" border="0" />
  </a>
</p>
<p align="center">
  Figure 3: Video Detection
</p>

```
python ADS_object_detection.py
```
<p align="center">
  <a href="https://www.youtube.com/watch?v=P13EDUTOlkg" target="_blank">
    <img src="http://img.youtube.com/vi/P13EDUTOlkg/0.jpg" alt="Description" width="480" height="360" border="0" />
  </a>
</p>
<p align="center">
  Figure 4: Object Detection for CARLA Driving Simulator by using YOLOv4
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=3gIghBNTxxQ" target="_blank">
    <img src="http://img.youtube.com/vi/3gIghBNTxxQ/0.jpg" alt="Description" width="480" height="360" border="0" />
  </a>
</p>
<p align="center">
  Figure 5: Object Detection for CARLA Driving Simulator by using YOLOv4 (path trajectory)
</p>

## Reference
https://github.com/AlexeyAB/darknet  
https://github.com/bubbliiiing/yolov4-pytorch  
[Introduction-Self-driving cars with Carla and Python](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/)  
https://github.com/DanielHfnr/Carla-Object-Detection-Dataset  