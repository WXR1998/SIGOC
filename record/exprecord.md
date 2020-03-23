# 实验记录

## exp00 : maskrcnn_baseline

删去sofa和BG两个类，剩余的类的错分现象不多。
400:    0.533 - 0.549
600:    0.543 - 0.560
1000:   0.556 - 0.574

## exp01 : exp00

添加新的网络结构：

需要收集整个数据集中，每对物体之间的相对位置关系分布

原结构：每个object proposal会输出一个label(fpn_classifier)
新结构：每个object proposal会输出一个label，将这个label和其feature map连接后

scale 100:  400: 0.517 - 0.533
scale 30:   400: 0.530 - 0.546
scale 20:   400: 0.531 - 0.547

## exp02 : exp01

修改了loss计算方法，原本是dist*coef，新的方法是1/(exp(dist)\*exp(coef))
scale 1:    400: 0.532 - 0.549
scale 2:    400: 0.538 - 0.554
scale 4:    400: 0.529 - 0.545

## exp03 : exp02

添加了loss可视化，便于查看rel分支的输出结果。
修改了loss计算的bug：对角线上元素loss异常

scale 2:    400: 0.538 - 0.554
scale 3:    400: 0.539 - 0.556
scale 4:    400: 0.532 - 0.550

## exp04 : exp03

加入了方向：将距离分为上下距离和前后左右距离。

scale 3:    400: 0.541 - 0.557
scale 3, 2.0: 400: 0.538 - 0.555

## exp05 : exp04

修改loss表达式。