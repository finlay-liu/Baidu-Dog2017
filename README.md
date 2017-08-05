## 说明

1. 解决思路

   使用各种Pre-trained的模型进行提取特征,并concat到一起,然后只用训练最后一层FC层,可复现线上0.173成绩.

   **TODO**:数据扩充和物体检测.

   代码使用方法:

   ```
   # 提取VGG13特征,并进行保存
   python torch_feature.py --model vgg13 --ffpath ./feature/vgg13.h5 

   # 提取DenseNet169特征,并进行保存
   python torch_feature.py --model densenet169 --ffpath ./feature/densenet169.h5

   # 使用提取的特征跑二层模型
   python torch_l2.py
   ```
   
2. 所用框架: Pytorch(拒绝keras和TensorSlow框架).

![](./feature/model.png)

## 其他

代码有问题请自己解决,谢谢,希望大家玩的开心!
