import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Setup
    print()
    alpha = .1 #[x, y, alpha]

    #Dataset
    try:
        dataset = np.genfromtxt("test.txt", delimiter=",")
    except:
        print("File not found")
        exit()

    print("Dataset: ")
    print(dataset)

    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    m = dataset.shape[0]
    weights = np.matrix([[-13, 7, 1, 1], [1, 1], [0,0,0]])
    bias = 1
    print(cost(weights, bias, dataset))

    print()
    print("X: ")
    print(x)

    print()
    print("Initial Weights:")
    print(weights)

    print()
    #print("Parameters:")
    param_size = (dataset.shape[1],1)
    param = np.ones(param_size)[:,0]
    cost(weights, bias, dataset)
    
#    print(weights[0,1][0]) #weights [0, layer][item]
    
#    print("Initial Cost: ", end='')



    for i in range(1,50000):
        weights = update(weights, bias, dataset, alpha)
        
    print(cost(weights, bias, dataset))
    print(test(weights, bias, dataset))
    print(weights)
        
"""
    print()
    print("Cost: ", end='')
    print(cost(weights, bias, dataset))

    print()
    print("Weights:")
    print(weights)

    print()
    print("[0, 0] -> ", end="")
    print(test(weights, bias, np.array([0,0])))

    print()
    print("[1, 0] -> ", end="")
    print(test(weights, bias, np.array([1,0])))

    print()
    print("[0, 1] -> ", end="")
    print(test(weights, bias, np.array([0,1])))

    print()
    print("[1, 1] -> ", end="")
    print(test(weights, bias, np.array([1,1])))
"""

def test(weights, bias, dataset):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-1)]
    m = dataset.shape[0]

    w_11 = weights[0,0][0]
    w_12 = weights[0,0][1]
    w_13 = weights[0,0][2]
    w_14 = weights[0,0][3]
    w_21 = weights[0,1][0]
    w_22 = weights[0,1][1]
    b_1 = weights[0,2][0]
    b_2 = weights[0,2][1]
    b_3 = weights[0,2][2]

    i_1 = x[:,0]
    i_2 = x[:,1]

    return sigmoid(w_21*sigmoid(w_11 *i_1 + w_12*i_2 + b_1) + w_22*sigmoid(w_13 *i_1 + w_14*i_2 + b_2) + b_3)

#Cost Function
def cost(weights, bias, dataset):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-1)]
    m = dataset.shape[0]

    print()

    w_11 = weights[0,0][0]
    w_12 = weights[0,0][1]
    w_13 = weights[0,0][2]
    w_14 = weights[0,0][3]
    w_21 = weights[0,1][0]
    w_22 = weights[0,1][1]
    b_1 = weights[0,2][0]
    b_2 = weights[0,2][1]
    b_3 = weights[0,2][2]

    i_1 = x[:,0]
    i_2 = x[:,1]

    return np.sum(np.square(y - sigmoid(w_21*sigmoid(w_11 *i_1 + w_12*i_2 + b_1) + w_22*sigmoid(w_13 *i_1 + w_14*i_2 + b_2) + b_3)))
     
def sigmoid(data):
    return 1/(1+np.exp( (-1) * data ))

def sigmoid_der(data):
    return ( np.exp( -1 * data ) ) / (np.square(1+np.exp( (-1) * data )))


def update(weights, bias, dataset, alpha):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-1)]
    m = np.shape(dataset)[0]

    w_11 = weights[0,0][0]
    w_12 = weights[0,0][1]
    w_13 = weights[0,0][2]
    w_14 = weights[0,0][3]
    w_21 = weights[0,1][0]
    w_22 = weights[0,1][1]
    b_1 = weights[0,2][0]
    b_2 = weights[0,2][1]
    b_3 = weights[0,2][2]

    i_1 = x[:,0]
    i_2 = x[:,1]

    A = sigmoid(w_11*i_1 + w_12*i_2)
    B = sigmoid(w_13*i_1 + w_14*i_2)
    
    shorten = sigmoid(w_21*A + w_22*B + b_3)*sigmoid_der(w_21*A + w_22*B + b_3)

    error = y - test(weights, bias, dataset)
    
    w_21 += np.sum(alpha * shorten*A*error)
    w_22 += np.sum(alpha * shorten*B*error)
    w_11 += np.sum(alpha * shorten * sigmoid_der( w_11*i_1 + w_12*i_2 + b_1)*i_1*error)
    w_12 += np.sum(alpha * shorten * sigmoid_der( w_11*i_1 + w_12*i_2 + b_1)*i_2*error)
    w_13 += np.sum(alpha * shorten * sigmoid_der( w_13*i_1 + w_14*i_2 + b_1)*i_1*error)
    w_14 += np.sum(alpha * shorten * sigmoid_der( w_13*i_1 + w_14*i_2 + b_1)*i_2*error)
    b_1 += np.sum(alpha * shorten * sigmoid_der( w_11*i_1 + w_12*i_2 + b_1)*error)
    b_2 += np.sum(alpha * shorten * sigmoid_der( w_11*i_1 + w_12*i_2 + b_1)*error)
    b_3 += np.sum(alpha *shorten*error)

    weights[0,0][0] = w_11
    weights[0,0][1] = w_12
    weights[0,0][2] = w_13
    weights[0,0][3] = w_14
    weights[0,1][0] = w_21
    weights[0,1][1] = w_22
    weights[0,2][0] = b_1
    weights[0,2][1] = b_2
    weights[0,2][2] = b_3

    
    
    return weights


main()
