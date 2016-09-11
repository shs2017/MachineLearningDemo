import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Setup
    print()
    alpha = .05
    
    #Dataset
    try:
        dataset = np.genfromtxt("test.txt", delimiter=",")
    except:
        print("File not found")
        exit()

    print("Dataset: ")
    print(dataset)

    param_size = np.shape(dataset)[1]
    param = np.ones(param_size) #[x, y]
    
    for i in range(1,2):
        newparam = update(param, dataset, alpha)
        param[0] = newparam[0]
        param[1] = newparam[1]

    print("Cost: ", end="n")
    print(cost(param, dataset))

    print("Slope: " + str(param[0]))
    print("Y-intercept: " + str(param[1]))

    #Graph
    print("Graphing...")
    plt.ion()
    print("Graph Bounds")
    minx = min(dataset[:,0])
    maxx = max(dataset[:,0])
    graph_param = [minx, maxx, min(dataset[:,1]),  max(dataset[:,1])]
    print(graph_param)
    print(param)
    print(param[0] * np.array(range(int(minx),int(maxx))) + param[1])
    plt.plot(param[0] * np.array(range(int(minx),int(maxx))) + param[1])
    plt.axis(graph_param)
    plt.show()
    plt.ioff()
    
#Cost Function
def cost(param, dataset):
    m = np.shape(dataset)[0]
    x = dataset[:,0]
    x=np.vstack((x, np.ones(m)))    
    y = dataset[:,1]
    a=param[0]
    b=param[1]
    return (1/(2*m))*np.sum(np.square(y-(np.dot(param, x))))

def update(param, dataset, alpha):
    m = np.shape(dataset)[0]
    a = param[0]
    b = param[1]
    x = dataset[:,0]
    x=np.vstack((x, np.ones(m)))
    y = dataset[:,1]
    newa = a + alpha*(1/m)*(np.sum((y-(np.dot([param], x)))*x))
    newb = b + alpha*(1/m)*(np.sum((y-(np.dot([param], x)))))
    return [newa, newb]

main()
