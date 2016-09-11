import numpy as np
import matplotlib as plt
import cv2
import os

def main():
    extrapolate_test()

def extrapolate_test():
    param = np.array([-4.08178328, 5.95933567, -2.62680833, -1.51356927]) #10x10 version

    print(param)


    img = cv2.imread('extrapolate/images.jpg')
    training = os.listdir("extrapolate")    
    for i in training:
        img = cv2.imread("extrapolate/" + i)

        shape = img.shape[0:2]
        scale = shape[0]/shape[1]
        res = cv2.resize(img, None, fx=((1/shape[1])*10) , fy=((1/shape[0])*10), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(("10x10/" + i), res)
        shape = res.shape[0:2]
        
        print("File: " + i)
        print("Shape: " + str(shape))

        

        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                abc = np.append(res[j,k],[1])
                if(np.round(1/(1+np.exp((np.dot([param],np.transpose(abc)))*-1)))):
                    res[j,k] = (255, 255, 255)
                else:
                    res[j,k] = (0, 0 ,0)
        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                try:
                    if(res[j,k] == res[j+1,k] and res[j,k] == res[j-1,k] and res[j,k] == res[j,k+1] and res[j,k] == res[j,k-1]):
                        res[j,k] = res[j,k+1]
                except:
                    pass
                        
                

                #a = (1/(1+np.exp((np.dot([param],np.transpose(abc)))*-1)))*255
                #print(a)
                #res[j,k] = (a, a, a)


        cv2.imwrite(("extrapolate-is-grass/" + i), res)
    
def test(): 
    param = np.array([-4.08178328, 5.95933567, -2.62680833, -1.51356927])
    print(param)


    training = os.listdir("original")    
    for i in training:
        img = cv2.imread("original/" + i)
        shape = img.shape[0:2]
        scale = shape[0]/shape[1]
        res = cv2.resize(img, None, fx=((1/shape[1])*10), fy=((1/shape[0])*10 ), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(("10x10/" + i), res)
        shape = res.shape[0:2]
        
        print("File: " + i)
        print("Shape: " + str(shape))
        

        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                abc = np.append(res[j,k],[1])
                if(np.round(1/(1+np.exp((np.dot([param],np.transpose(abc)))*-1)))):
                    res[j,k] = (255, 255, 255)
                else:
                    res[j,k] = (0, 0 ,0)
        cv2.imwrite(("is-grass/" + i), res)

    for i in range(0,500):
        print("hi")

                    
    training = os.listdir("original-not")    
    for i in training:
        img = cv2.imread("original-not/" + i)
        shape = img.shape[0:2]
        scale = shape[0]/shape[1]
        res = cv2.resize(img, None, fx=((1/shape[1])*10), fy=((1/shape[0])*10 ), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(("10x10/" + i), res)
        shape = res.shape[0:2]
        
        print("File: " + i)
        print("Shape: " + str(shape))
        

        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                abc = np.append(res[j,k],[1])
                if(np.round(1/(1+np.exp((np.dot([param],np.transpose(abc)))*-1)))):
                    res[j,k] = (255, 255, 255)

                else:
                    res[j,k] = (0, 0 ,0)

                                                
        cv2.imwrite(("is-grass/" + i), res)

                
def train():
    #Setup
    print()
    alpha = .05 #[x, y, alpha]

    #Dataset
    try:
        dataset = np.genfromtxt("dataset.txt", delimiter=",")
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


def normalize():
    training = os.listdir("original")
    f = open("dataset.txt", "w")

    for i in training:
        img = cv2.imread("original/" + i)
        shape = img.shape[0:2]
        scale = shape[0]/shape[1]
        res = cv2.resize(img, None, fx=((1/shape[1])*20), fy=((1/shape[0])*20 ), interpolation = cv2.INTER_NEAREST)
        print(res.shape)
        res1 = cv2.resize(res, None, fx=10, fy=10, interpolation= cv2.INTER_NEAREST)

        shape = res.shape[0:2]
        
        print("File: " + i)
        print("Shape: " + str(shape))
        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                color = ', '.join(map(str, res[j,k]))
                f.write(str(color) + ", 1\n")     

        cv2.imwrite(("10x10/" + i), res1)
        
    training = os.listdir("original-not")
    for i in training:
        img = cv2.imread("original-not/" + i)
        shape = img.shape[0:2]
        print(shape)
        scale = shape[0]/shape[1]
        res = cv2.resize(img, None, fx=((1/shape[1])*20), fy=((1/shape[0])*20 ), interpolation = cv2.INTER_NEAREST)
        print(res.shape)
        res1 = cv2.resize(res, None, fx=10, fy=10, interpolation= cv2.INTER_NEAREST)

        shape = res.shape[0:2]
        
        print("File: " + i)
        print("Shape: " + str(shape))
        for j in range(1,shape[0]):
            for k in range(1,shape[1]):
                color = ', '.join(map(str, res[j,k]))
                f.write(str(color) + ", 0\n")     

        cv2.imwrite(("10x10-not/" + i), res1)
    f.close()        
    
main()

#cv2.imshow('WAVY', res1)
#cv2.waitKey(0)
