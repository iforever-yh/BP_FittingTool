import numpy as np
import math
import matplotlib.pyplot as plt

#useful functions 有用的函数
#fun1 安置对角元素
def list2diagMatrix(diag):
    col = len(diag)
    result = np.zeros((col, col))
    for i in range(col):
        for j in range(col):
            if i == j:
                result[i][j] = diag[i]

    return result

#layer class 层类
class layer:
    def __init__(self, layerName='hidden', transferFunction='logsig'):
        self.layerInfo = {
            'layerName':layerName,
            'transferFunction':transferFunction,
        }
    
    #添加输入层接口
    def addFirstLayer(self, row, inputShape):
        self.layerWight = (np.random.rand(row, inputShape) - 0.5) * 2.0
        self.layerBias = (np.random.rand(row, 1) - 0.5) * 2.0
        self.layerInfo['row(num_of_cells_in_this_layer)'] = row
        self.layerInfo['col(inputs_for_each_cell)'] = inputShape

    #添加隐藏层
    def addHiddenLayer(self, row, col):
        self.layerWight = (np.random.rand(row,col) - 0.5) * 2.0
        self.layerBias = (np.random.rand(row, 1) - 0.5) * 2.0
        self.layerInfo['row(num_of_cells_in_this_layer)'] = row
        self.layerInfo['col(inputs_for_each_cell)'] = col

#NN class 网络类
class neuralNetworkModel:
    def __init__(self, learningRate = 0.1):
        self.learningRate = learningRate
        self.layersNum = 0
        self.layerInfo = []
        self.wight = []
        self.bias = []

    #添加新层
    def addLayer(self, row=1, inputShape=[], transferFunction='logsig'):
        if len(inputShape) != 0:
            layerAdded = layer(layerName='firstLayer', transferFunction=transferFunction)
            layerAdded.addFirstLayer(row=row, inputShape=inputShape[0])
            self.wight.append(layerAdded.layerWight)
            self.bias.append(layerAdded.layerBias)
            self.layerInfo.append(layerAdded.layerInfo)
        else:
            if self.layersNum == 0:
                print('Input set error, add a layer after inoput layer!!!')
                exit(3)
            else:
                col = self.wight[-1].shape[0]
                layerAdded = layer(layerName='hiddenLayer', transferFunction=transferFunction)
                layerAdded.addHiddenLayer(row=row, col=col)
                self.wight.append(layerAdded.layerWight)
                self.bias.append(layerAdded.layerBias)
                self.layerInfo.append(layerAdded.layerInfo)
        self.layersNum += 1

    #正向传播
    def forward(self, input):
        layerOut = np.array([input]).T
        self.input = layerOut
        self.layerOutHistory = []
        for i_iter in range(self.layersNum):
            n = np.dot(self.wight[i_iter], layerOut)
            n = np.add(n, self.bias[i_iter])
            if self.layerInfo[i_iter]['transferFunction'] == 'logsig':
                layerOut = np.array([[1.0 / (1.0 + math.exp(-i_sub1)) for i_sub1 in n]]).T
                self.layerOutHistory.append(layerOut)
            elif self.layerInfo[i_iter]['transferFunction'] == 'pruelin':
                layerOut = np.array([i_sub2 for i_sub2 in n]).T
                self.layerOutHistory.append(layerOut)

        return layerOut

    #损失计算
    def loss(self, predict_value, target_value):
        e = target_value - predict_value
        return e.T

    #传函导数计算
    def get_dTransferFunctions_dn(self, layer_id, funName='pruelin'):
        if funName == 'logsig':
            df_dn_diag = [(1 - i_sub1) * (i_sub1) for i_sub1 in self.layerOutHistory[layer_id]]
            df_dn = list2diagMatrix(df_dn_diag)
        elif funName == 'pruelin':
            df_dn_diag = [1] * self.layerInfo[-1]['row(num_of_cells_in_this_layer)']
            df_dn = list2diagMatrix(df_dn_diag)

        return df_dn

    #反向传播
    def backward(self, e):
        f = [1] * self.layersNum
        s = [1] * self.layersNum

        i_iter = self.layersNum - 1
        while i_iter + 1:
            f[i_iter] = self.get_dTransferFunctions_dn(i_iter, funName=self.layerInfo[i_iter]['transferFunction'])
            i_iter -= 1

        s[-1] = -2 * np.dot(f[-1], e)
        i_iter = self.layersNum - 2
        while i_iter + 1:
            s[i_iter] = np.dot(np.dot(f[i_iter], (self.wight[i_iter + 1]).T), s[i_iter + 1])
            i_iter -= 1

        i_iter = self.layersNum - 1
        while i_iter:
            self.wight[i_iter] = self.wight[i_iter] - np.dot(self.learningRate * s[i_iter], self.layerOutHistory[i_iter - 1].T)
            self.bias[i_iter] = self.bias[i_iter] - self.learningRate * s[i_iter]
            i_iter -= 1
        
        self.wight[0] = self.wight[0] - np.dot(self.learningRate * s[0], np.array(self.input).T)
        self.bias[0] = self.bias[0] - self.learningRate * s[0]

    #输出网络信息（给文件名就从文件输出，不给就在终端输出）
    def summary(self, fout_name=''):
        if fout_name != '':
            with open(fout_name,'w') as f_output:#按指定格式输出txt文档
                f_output.write('layerInfo')
                for i in range(self.layersNum):
                    f_output.write('\n' + 'layer: ' + str(i + 1) + '\n')
                    f_output.write('row: ' + str(self.layerInfo[i]['row(num_of_cells_in_this_layer)']) + '\t')
                    f_output.write('col: ' + str(self.layerInfo[i]['col(inputs_for_each_cell)']) + '\n')
                    f_output.write('transferFunction: ' + self.layerInfo[i]['transferFunction'])
                print('Write layerInfo finished...')
        else:
            print('layerInfo')
            for i in range(self.layersNum):
                print('\n' + 'layer: ' + str(i + 1))
                print('row: ' + str(self.layerInfo[i]['row(num_of_cells_in_this_layer)']))
                print('col: ' + str(self.layerInfo[i]['col(inputs_for_each_cell)']))
                print('transferFunction: ' + self.layerInfo[i]['transferFunction'])

    #主程序控制接口
    def netTrain(self, input_x, input_y, sub_epoch=10, showLoss=False):
        i_iter = 1
        while i_iter <= sub_epoch:
            layerOut = self.forward(input=input_x)
            e = self.loss(predict_value=layerOut, target_value=np.array([input_y]))
            if showLoss:
                print(f'Epoch: {i_iter}\tloss: {e.T[0]}')
            self.backward(e=e)
            i_iter += 1
    
    #可视化，若要对比原函数则要输入其值
    def netShow(self, min=0, max=1, step=0.5, compare=False, x_compare=[], y_compare=[]):
        x_values = np.arange(min, max, step)
        y_values = [self.forward([x])[0] for x in x_values]
        plt.style.use('seaborn')

        fig, ax = plt.subplots()
        ax.plot(x_values, y_values, 'b', linewidth=3, label='Fitted')
        if compare:
            ax.plot(x_compare, y_compare, 'r', linewidth=3, label='Original')
            print(y_values)
        ax.legend()

        #图标信息
        ax.set_title('Results', fontsize=24)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)

        #坐标比例
        ax.tick_params(axis='both', labelsize=14)
        plt.show()
