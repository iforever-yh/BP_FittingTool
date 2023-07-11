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
    x, y=tsg.genpower(-3,3,0.1,3)
    main(input_x=x, input_y=y, epoch=10000)
    print('\nProgram over...\n')
    os.system('pause')
    exit(0)
