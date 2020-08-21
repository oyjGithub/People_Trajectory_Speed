# **通过opencv进行行人方向和速度估计**
## 目标：分析行人2d box数据，估计出的行人当前运动方向和大概速度，并在视频上可视化出来。
### 1.导入包multiprocessing，numpy，argparse，dlib，cv2
1）.采用预训练模型mobilenet_ssd，这是一种适用于移动端而提出的一种轻量级深度网络模型。
主要使用了深度可分离卷积Depthwise Separable Convolution将标准卷积核进行分解计算，减少了计算量。保证了检测速度。
将图片进行归一化处理，在进行检测。
2）.utils文件做了一个帧率的计算。
3）.模型过滤了除了“人”之外的其它类别，减少了box数量
4）.对每个box计算其置信度，如果低于设定的0.3，则过滤掉。
5）.设定在每30帧进行一次box框检测，为了保持较高的检测准确率，并且释放之前的内存。
6）.统计每个box框的坐标点，绘轨迹，和计算场景内行人每一步的速度。
### 2.运行效果图
图片中的黑点表示该时间步的轨迹点
![图1](https://github.com/onlylove321/People_Trajectory_Speed/blob/master/test1.jpg)
![图1](https://github.com/onlylove321/People_Trajectory_Speed/blob/master/test2.jpg)
![图1](https://github.com/onlylove321/People_Trajectory_Speed/blob/master/test3.jpg)
***
### 3.相关代码块
速度计算，图片帧率0.14s
```      
c, d = circlelist[i-len(inputQueues)]
distance = float(((c - endX)**2 + (d - endY)**2)**(0.5))
print(c,d,endX, endY)
fact_distance = distance / 300
speed = fact_distance / 0.14
```
### 4.检测视频
test2.mp4

### 5.输出视频
demo.mp4