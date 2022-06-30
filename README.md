# BP_FittingTool
构建神经网络，使用BP（反向传播算法）拟合函数。

Build a Neural Network to fit functions with the Backpropagation Algorithm.

# Libs you need
numpy, matplotlib

windows平台可在cmd中使用下面的命令安装库。

for windows you can use these codes in cmd to install libs.
(```
pip install --user numpy

pip install --user matplotlib
```)

# How to use
（1）首先在BP_main.py的main()中创建网络对象，对于的代码如下：
(```
network = nnm.neuralNetworkModel(learningRate = 0.05)     # 定义网络对象，可设置参数为学习率

network.addLayer(8, inputShape=[1])                       # 添加隐藏层，第一个参数为该层神经元数，第二个参数为输入训练集的规模，传递函数目前只有log-sigmoid

network.addLayer(8)                                       # 增加隐藏层，传递函数目前只有log-sigmoid

network.addLayer(1, transferFunction='pruelin')           # 设置输出层，第一个参数为输出参数规模，第二个为传递函数，目前只支持pruelin
```)

（2）设置训练集，在BP_main.py中的if __name__ == '__main__':
可使用自带的训练集生成函数，也可以
x, y=tsg.genpower(-3,3,0.1,3)
