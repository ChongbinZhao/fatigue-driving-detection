### 提示：该README.md所用到的图片都保存在src文件夹里，图片没法加载的话可能需要你先挂一下vpn！
<br></br>


### 疲劳驾驶检测系统（demo）
<br>


#### 1.实际效果

##### 1.1 系统界面

<img src="https://raw.githubusercontent.com/ChongbinZhao/fatigue-driving-detection/master/pic/2.png" alt="img" style="zoom: 50%;" />
<br>


##### 1.2 全部项目文件

由于项目里有一些比较大的库文件以及封装后的exe比较大，github无法上传，所以完整版[点击这里](https://pan.baidu.com/s/17ehBuduMEbJF8OS6q9IWWQ?pwd=3456)下载（提取码3456）

<img src="https://raw.githubusercontent.com/ChongbinZhao/fatigue-driving-detection/master/pic/1.png" alt="img" style="zoom: 67%;" />
<br>

#### 2.基本介绍

- 利用驾驶人的`眼动特性`、`嘴部运动特性`和`头部运动特性`等推断驾驶人的疲劳状态
- 功能介绍：
  - 具有良好交互界面
  - 驾驶人疲劳信息实时反馈
  - 检测效果稳定
  - 可通过交互界面手动调整算法参数 
  - 具有声响警报功能
<br>


#### 3.环境配置

##### 3.1 系统环境

- Windows10
- Python3.6.4
- Pycharm Community 2021
<br>
 

##### 3.2 主要的第三方库

- dlib：一个非常经典的人脸识别和人脸特征提取框架，程序中对于人脸疲劳特征的检测都是基于这个框架来进行的
- Opencv(cv2)：负责绝大部分的图像处理操作
- numpy：数据处理
- math：基本的数学运算
- thread：用于建立多线程，使得主程序和子程序能够独立运行且不冲突
- wx(wxbuilder)：一个基于Python、用于构造UI界面的库
- pyinstaller：将py脚本文件封装成可执行exe文件
<br>




#### 4.检测原理

具体检测原理参考博客（此项目在此博客的基础上进行拓展，增加了`疲劳信息重置`、`疲劳警报`和`软件封装`等功能，并且搜集了一些博主没有给出的库文件，例如`shape_predictor_68_face_landmarks.dat`）

[Dlib模型之驾驶员疲劳检测一](https://cungudafa.blog.csdn.net/article/details/103477960)

[Dlib模型之驾驶员疲劳检测二](https://cungudafa.blog.csdn.net/article/details/103496881)

[Dlib模型之驾驶员疲劳检测三](https://cungudafa.blog.csdn.net/article/details/103499230)
<br>


##### 4.1 获取人脸模型的68个特征点

- 首先先用人脸检测器dlib.get_frontal_face_detector()将人脸部分检测出来，然后用矩形框标出
- 紧接着用dlib.shape_predictor()函数和人脸识别开源数据集shape_predictor_68_face_landmarks.dat识别出人脸的68个特征点，然后在每一帧图片上标注出来

<img src="https://raw.githubusercontent.com/ChongbinZhao/fatigue-driving-detection/master/pic/3.png" alt="img" style="zoom: 67%;" />
<br>


##### 4.2 检测驾驶人眨眼情况

当眼睛闭合到一定程度（EAR小于给定阈值）并持续数秒，就可以判断驾驶人为疲劳驾驶，其中眼睛特征点分布如下

<img src="https://raw.githubusercontent.com/ChongbinZhao/fatigue-driving-detection/master/pic/4.png" alt="img" style="zoom: 67%;" />
<br>




##### 4.3 检测驾驶人打哈欠情况

定义嘴部宽度比`MAR：Mouth Aspect Ratio`，MAR计算原理与EAR计算原理一致
<br>


##### 4.4 检测驾驶人瞌睡点头情况

这一部分比较难懂，涉及到的知识有`头部姿态估计`和`像素坐标系与空间坐标系之间的转换`
<br>


##### 4.5 建立多线程

- 为了将疲劳驾驶检测部分和警报播报部分能异步运行，我们需要用到thread库来建立多线程
- 如果不建立多线程，则只有警报播报完之后才能继续进行疲劳驾驶检测
- 另外，在程序中多次启用同一个线程，每一次都要重新将这个线程实例化
<br>


#### 5. UI界面

##### 5.1 基本工具

- 此项目所构造的UI界面基于第三方库wx，该第三方库能够提供C++、Python、PHP、Lua、XRC等不同语言的API，项目采用的是Python语言
- UI界面设计的核心功能是`信号和槽`，即一个控件可以直接触发一个函数（当然函数中可以包含其他函数）
<br>


##### 5.2 界面布局与主要控件功能

<img src="https://raw.githubusercontent.com/ChongbinZhao/fatigue-driving-detection/master/pic/2.png" alt="img" style="zoom: 50%;" />
<br>


- **摄像头ID**：可控选择的有摄像头ID_0和摄像头ID_1，当计算机不接外置摄像头时，摄像头ID_0默认是本地摄像头；当接入外置摄像头时，可以通过切换摄像头ID来选择想要调用的摄像头。



- **打开摄像头**：打开与摄像头ID相对应的摄像头。



- **打开视频文件**：程序中不仅可以实时检测驾驶人当前的疲劳情况，也可以用提前录制好的视频文件来进行检测。



- **停止**：暂停检测并且之后重新检测时会将已记录的疲劳信息置零。



- **疲劳检测**：在对应的选项打勾，程序就会自动对打勾的选项进行检测。每一个检测项目的帧数阈值都可以通过列表框手动改变。



- **脱岗检测**：当程序检测不到驾驶人脸特征时，说明驾驶人脱岗，有疲劳驾驶或危险驾驶的倾向，这时候会在状态检测栏输出脱岗警告。



- **帧数阈值**：用眨眼检测来举例，假如当前对应的帧数阈值我们设置为20，则当摄像头所拍摄到的连续20帧驾驶人眼睛宽度比EAR都小于self.HAR_THRESH时（self.HAR_THRESH可以手动调整，但通常为0.35），self.eTOTAL加1（self.HAR_THRESH可以手动调整，但为了方便调试，此程序中暂定为4），当self.eTOTAL大于4时，程序就会启动警报播放线程。



- **信息重置**：信息重置可以手动将之前已保存的疲劳信息置零。



- **关闭警报**：关闭警报则是负责销毁已启用的警报播放线程，因为关闭警报播放线程实质上是将这个线程销毁，所以再次启用警报播放线程时要重新将这个线程实例化，否则程序会报错。
<br>


#### 6. 封装成可执行exe

将Python脚本文件封装成可执行exe用的是`Pyinstaller`，下载工程文件后点击`疲劳驾驶检测程序.exe`就可以直接运行程序。
<br>
