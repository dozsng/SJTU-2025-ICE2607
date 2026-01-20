4.1 Pytorch

代码位于 code/Pytorch 中，进入 code/Pytorch 目录下：

final.pth为最终模型，test_acc.png为test acc变化趋势，

运行exp2.py即可测试主代码，可能需运行5~10分钟， resnet20模型在 models.py 中，预训练了5轮的pretrain_model.pth在文件夹pretrained中。

4.2 CNN

代码位于 code/CNN 中，进入 code/CNN 目录下：

运行main.py即可测试主代码，输入数据集图像位于 CNN/Dataset 中，输入待检索图像位于 CNN/Query_image 中，

输出文件夹分别为：

CNN/Feature_xxx --- 使用 xxx 模型得到的数据集图像特征；
CNN/Query_feature --- 待检索图像的特征；
CNN/Output --- 检索结果。
