import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Setup
    print()
    alpha = .05 #[x, y, alpha]

    #Dataset
    try:
        dataset = np.genfromtxt("test.txt", delimiter=",")
    except:
        print("File not found")
        exit()

    print("Dataset: ")
    print(dataset)

    x = dataset[:,0]
    m = dataset.shape[0]
    print()
    #print("Parameters:")
    param_size = (dataset.shape[1],1)
    param = np.ones(param_size)[:,0]
     
    print("Initial Param: ", end="")
    print(param)

    print("Initial Cost: ", end='')
    print(cost(param, dataset))

    print("Initial Predictions:")
    print(np.round(sigmoid(param,dataset)))

    for i in range(1,1000):
        param = update(param,dataset,alpha)[0,:]
        
    #Graph
    #print("Graphing...")
    #plt.ion()
    #print("Graph Bounds")
    minx = min(dataset[:,0])
    maxx = max(dataset[:,0])
    graph_param = [minx, maxx, min(dataset[:,1]),  max(dataset[:,1])]
    #print(graph_param)
    print()
    print("Final Param:", end="")
    print(param)
    print("Final Cost:", end="")
    print(cost(param,dataset))
    print("Rounded Predictions: ", end="")
    print(np.round(sigmoid(param,dataset)))
    #plt.plot(param[0] * np.array(range(int(minx),int(maxx))) + param[1])
    #plt.axis(graph_param)
    #plt.show()
    #plt.ioff()
  

#Cost Function
def cost(param, dataset):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-1)]
    m = dataset.shape[0]
    prob = (sigmoid(param, dataset))
    return (1/(2*m)) * np.square(np.sum((y)*(np.log(prob)) + (1-y)*(np.log(1-prob))))
    
def sigmoid(param, dataset):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    m = dataset.shape[0]
    x = np.hstack((x, np.transpose([np.ones(m)])))
    return 1/(1+np.exp((np.dot([param],np.transpose(x)))*-1))


def update(param, dataset, alpha):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-1)]
    m = np.shape(dataset)[0]
    return_param= np.zeros((1,n))
    for i in range(0,(n-1)):
        return_param[0,i] = param[i] + alpha*(1/m)*(np.sum((y -  sigmoid(param, dataset))*x[:,i]))
    return_param[0,n-1] = param[(n-1)] + alpha*(1/m)*(np.sum(y - (sigmoid(param, dataset))))
    return return_param


main()
