# BP_FittingTool
构建神经网络，使用BP（反向传播算法）拟合函数。  
Build a Neural Network to fit functions using the Backpropagation Algorithm.

# Requirements 需要安装的库
numpy, matplotlib

windows平台可在cmd中使用下面的命令安装库。  
for windows you can use these command to install the libs in cmd.
```
pip install --user numpy  
pip install --user matplotlib
```

# How to use 使用方法
1.在BP_main.py的main()中创建网络对象，对于的代码如下：
```
network = nnm.neuralNetworkModel(learningRate = 0.05)     # 定义网络对象，可设置参数为学习率  
network.addLayer(8, inputShape=[1])                       # 添加隐藏层，第一个参数为该层神经元数，第二个参数为输入训练集的规模，传递函数目前只有log-sigmoid  
network.addLayer(8)                                       # 增加隐藏层，传递函数目前只有log-sigmoid  
network.addLayer(1, transferFunction='pruelin')           # 设置输出层，第一个参数为输出参数规模，第二个为传递函数，目前只支持pruelin
```
2.设置训练集，在BP_main.py中的if __name__ == '__main__':
+ 可使用自带的训练集生成函数
```
x, y=tsg.genpower(-3,3,0.1,3)                             # 参数格式(min, max, step)
```
+ 也可以自定义输入（输入要求为list）

3.调用main()执行：
```
main(input_x=x, input_y=y, epoch=10000)
```

# Result 结果展示
在main()函数中调用类的函数即可。
```
network.netShow(min=0, max=6, step=0.2, compare=True, x_compare=input_x, y_compare=input_y)  # 前三个为验证集参数，compara确定是否余目标函数对比（需提供最后两个参数）  
network.summary('outfile.txt')  # 将网络信息输出到文本
```
