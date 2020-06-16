说明：
本赛题任务主要分为两个子任务：目标点定位和图像分类
目标点定位：
需要从给定的图像中找到椎体与椎间盘所在的位置，尽量靠近目标的中间位置，因为后面的图像分类是以该点为中心裁剪的32×32大小的patch，进行分类。
总共有201个study，所以只需要返回201个上述这样的检测结果即可。就是检测出椎体和椎间盘的位置，需要根据下图对应好当前检测的椎体属于哪一段。
pipeline如下：
![Image](https://github.com/jzjbycfg/20200615/tree/master/tmp/pipeline.png)
![Image](https://github.com/jzjbycfg/20200615/blob/master/tmp/Figure_4.png)
