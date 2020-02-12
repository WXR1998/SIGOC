# 实验记录

## exp00 : maskrcnn_baseline

删去sofa和BG两个类，剩余的类的错分现象不多。
0.533 - 0.549

## exp01 : exp00

添加新的网络结构：

需要收集整个数据集中，每对物体之间的相对位置关系分布

原结构：每个object proposal会输出一个label
新结构：每个object proposal会输出一个label，将这个label和其feature map连接后
