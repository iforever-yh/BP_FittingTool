import os
import neuralNetworkModel as nnm
import trainSetGen as tsg

#useful function 
#fun1 check input lists 输入检查
def data_check(x, y):
    if x and y:
        if len(x) == len(y):
            print('\ncheck finished...\n')
        else:
            print('\nx and y are not matched!!!\n')
            exit(2)
    else:
        print('\nx or y is empty!!!\n')
        exit(1)

#main function 主控
def main(input_x=[], input_y=[], epoch=1000):
    data_check(input_x, input_y)    #check input lists
    network = nnm.neuralNetworkModel(learningRate = 0.05)
    network.addLayer(8, inputShape=[1])
    network.addLayer(8)
    network.addLayer(1, transferFunction='pruelin')

    epochOrigin = epoch
    while epoch:
        for i in range(len(input_x)):
            network.netTrain(input_x=input_x[i], input_y=input_y[i], sub_epoch=25)
        os.system("cls")
        print(f'Training {round(100 * (1 - epoch / epochOrigin), 2)}% ...')
        epoch -= 1

    network.netShow(min=0, max=6, step=0.2, compare=True, x_compare=input_x, y_compare=input_y)
    network.summary('outfile.txt')
    
#手动参数控制
if __name__ == '__main__':
    #x, y=tsg.genpower(-3,3,0.1,3)
    x = [[1],[1.2],[1.4],[1.6],[1.8],[2],[2.2],[2.4],[2.6],[2.8],[3.0],[3.2],[3.4],[3.6],[3.8],[4.0],[4.2],[4.4]]
    y = [[1.5],[0.5376],[-0.0429],[-0.3569],[-0.4920],[-0.5125],[-0.4649],[-0.3814],[-0.2840],[-0.1871],[0.1],[-0.0287],[0.0227],[0.0515],[0.0564],[0.0375],[-0.002],[-0.0547]]
    main(input_x=x, input_y=y, epoch=10000)
    print('\nProgram over...\n')
    os.system('pause')
    exit(0)
