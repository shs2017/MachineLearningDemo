from __future__ import print_function
import json
import numbers
import collections
import string
import ast
import numpy as np
from pprint import pprint
import itertools
from collections import defaultdict

#***TODO: Refactor code so that for i in range(0, len(...) is in form for i in ...
#***TODO: Change [:] to [] when possible so entire new copies wont have to be created all the time
#For ANN where inputs start at different places or jump layer just use a placeholder called "placeholder" so that the code doesn't have to recheck everything everytime and just know to skip it and carry result
def main():
    print()
#    with open('config2.json') as data_file:
    with open('configuration.json') as data_file:
        data = json.load(data_file)
    pprint(data)
    input = data["inputs"]
    input = ast.literal_eval(input)
    inputLen = len(input)
    layers = data["layers"]
    print("layers")
    print(layers)
    print()
    print("Inputs:")
    for i in range(0,len(input)):
        print(input[i][len(input[i])-1])
        print(string.join(input[i], sep="->"))
        print(getValue(data, input[i]))
        print()

    abc = 4



#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[1]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [1, 1, 1, 1, 1, 1, 1, 1, 1]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[2]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [2, 2, 2, 2, 2, 2, 2, 2, 2]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[3]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [3, 3, 3, 3, 3, 3, 3, 3, 3]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[4]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [4, 4, 4, 4, 4, 4, 4, 4, 4]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[5]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [5, 5, 5, 5, 5, 5, 5, 5, 5]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[6]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [6, 6, 6, 6, 6, 6, 6, 6, 6]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[7]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [7, 7, 7, 7, 7, 7, 7, 7, 7]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[8]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [8, 8, 8, 8, 8, 8, 8, 8, 8]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[9]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [9, 9, 9, 9, 9, 9, 9, 9, 9]))




    alpha = .075
#    print()


    #Dataset
    try:
        dataset = np.genfromtxt("testset.txt", delimiter=",")
    except:
        print("File not found")
        exit()

    print("Dataset: ")
    print(dataset)
    n = dataset.shape[1] #Columns in dataset
    m = dataset.shape[0] # Rows in dataset
    #Path was eliminated by computeLayer function - fix this - in mean time just get data back from file
    input = data["inputs"]
    input = ast.literal_eval(input)

    pprint(input)


    for i in range(0, 2000):
              data = updateWeights(dataset, data, input, alpha)
#    data = updateWeights(dataset, data, input, alpha)

    pprint(data)



#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[1]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [1, 1, 1, 1, 1, 1, 1, 1, 1]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[2]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [2, 2, 2, 2, 2, 2, 2, 2, 2]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[3]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [3, 3, 3, 3, 3, 3, 3, 3, 3]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[4]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [4, 4, 4, 4, 4, 4, 4, 4, 4]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[5]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [5, 5, 5, 5, 5, 5, 5, 5, 5]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[6]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [6, 6, 6, 6, 6, 6, 6, 6, 6]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[7]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [7, 7, 7, 7, 7, 7, 7, 7, 7]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[8]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [8, 8, 8, 8, 8, 8, 8, 8, 8]))
#    input = data["inputs"]
#    input = ast.literal_eval(input)
#    print("[9]")
#    print("LAYER ", end="")
#    print(abc, end=": ")
#    print(computeLayer(data, input, abc, [9, 9, 9, 9, 9, 9, 9, 9, 9]))


    return



"""
    input = data["inputs"]
    input = ast.literal_eval(input)
    print("[0, 0]")
    print("LAYER ", end="")
    print(abc, end=": ")
    print(computeLayer(data, input, abc, [0, 0]))

    input = data["inputs"]
    input = ast.literal_eval(input)
    print("[0, 1]")
    print("LAYER ", end="")
    print(abc, end=": ")
    print(computeLayer(data, input, abc, [0, 1]))

    input = data["inputs"]
    input = ast.literal_eval(input)
    print("[1, 0]")
    print("LAYER ", end="")
    print(abc, end=": ")
    print(computeLayer(data, input, abc, [1, 0]))

    input = data["inputs"]
    input = ast.literal_eval(input)
    print("[1, 1]")
    print("LAYER ", end="")
    print(abc, end=": ")
    print(computeLayer(data, input, abc, [1, 1]))
    return
"""



def getValue(data, path):
    for i in range(0,len(path)):
        data = data[path[i]]
    return data


#Todo: Check if Error Computes Right
def Error(dataset, data, path, outputLength):
    #Fix this to work with numpy instead of loop
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    y = dataset[:,(n-outputLength):n] # Fix to work with output length
    m = dataset.shape[0]
    x_n = x.shape[1]
    x_m = x.shape[0]
    output = []
    numLayer = data["layers"]-1

    for i in range(0, x_m):
        path = data["inputs"]
        path = ast.literal_eval(path)
        output.append(computeLayer(data, path[:], numLayer, x[i]))
    return y - output



#def setWeights(data, weightPath, values):
#    for key in weightPath[:-1]:
#        data = data.setdefault(key, {})
#    data[weightPath[-1]] = values
#    return data

#Updates the weights and Bias
#Assumes at least three layer network
def updateWeights(dataset, data, path, alpha):
    n = dataset.shape[1]
    x = dataset[:,0:(n-1)]
    m = dataset.shape[0]
    x_n = x.shape[1]
    x_m = x.shape[0]

    repeatWeights = []
    weightPaths = []
    weights = []

    repeatBias = []
    biasPaths = []
    bias = []
    computeBias = []
    B = []

    repeatLayer = []
    layer = []
    numLayer = 0
    l = []

    input = []
    I = []
    repeatInput = []

    error = Error(dataset, data, path, 1) #1 is output Length -- make a variable to know how many columns y takes up and replace 1 with that

    for i in range(0, len(path)):
        for j in range(0, len(path[i])):
            #Weights
            if path[i][j][0] == 'w' and (path[i][j] in repeatWeights) == False:
                #Get Value of Weights
                tmpWeight = path[i][:j+1]
                tmpWeight.append("value")
                weights.append(getFromDict(data, tmpWeight))
                #Get Path of Weights
                weightPaths.append(path[i][:j+1])
                #Keep track of Weights already done
                repeatWeights.append(path[i][j])
            #Bias and Compute
            if path[i][j] == 'sigmoid':
                #Bias -- Get The Paths Need
                if (path[i][:j+1] in repeatBias) == False:
                    tmpcomputeBias = []
                    tmpB = []
                    #Get Value of Bias
                    tmpBias = path[i][:j+1]
                    tmpBias.append("bias")
                    bias.append(getFromDict(data, tmpBias))
                    #Get Path of Weights
                    biasPaths.append(tmpBias)
                    #Keep track of Bias already done
                    repeatBias.append(path[i][:j+1])
                    computeBias.append(path[i][:j+1])
                #Compute
                if (len(path[i][:j+1]) in repeatLayer) == False:
                    tmpLayer = []
                    tmpL = []
                    length = len(path[i])-j-1
                    for k in range(0, x_m):
                        inputs = ast.literal_eval(data["inputs"])
                        tmpLayer.append(list(computeLayer(data, inputs, length, x[k], True)))
                        inputs = ast.literal_eval(data["inputs"])
                        tmpL.append(list(computeLayer(data, inputs, length, x[k])))
                    repeatLayer.append(len(path[i][:j+1]))
                    layer.append(np.matrix(tmpLayer))
                    l.append(np.matrix(tmpL))
                    numLayer += 1
            if path[i][j][:5] == 'input' and (path[i][:j+1] in repeatInput) == False:
                tmpInput = []
                tmpI = []
                length = len(path[i])-j-1
                for k in range(0, x_m):
                    inputs = ast.literal_eval(data["inputs"])
                    tmpInput.append(list(computeLayer(data, inputs, length, x[k])))
                repeatInput.append(path[i][:j+1])
                input.append(np.matrix(tmpInput))

    inputLayer = []
    #Group computeBias
#    for i in range(0, len(computeBias)):
#        for j in range(i+1, len(computeBias)):
#            if len(computeBias[i]) == len(computeBias[j]):
#                computeBias[i] = [computeBias[i], computeBias[j]]
#                computeBias.pop(j)


#    print("LAYER 1")
#    print(layer)

    for i in range(0, numLayer):
        #For Layers below last multiply current layer, and layer before (which should itself be multiplied by error, current layer, and layer before ... until it is multiplied by first layer which was originally multiplied by error and alpha)
        #First Layer
        if i == numLayer-1:
#            print("LAYER[i]")
#            print(np.array(layer[i]))
#            print(np.array(layer[i])[:,0:2])
#            print("LAYER[i-1]")
#            print(np.array(layer[i-1]))


#            layer[i] = np.matrix(np.array(layer[i]) * np.array(layer[i-1])) #KEEP THIS
#DELTE THIS TO
            notlayer = layer[:]
#            print(layer)
            for j in range(0, layer[i-1].shape[1]):
                blah = layer[i].shape[1]
                notlayer[i][:,(j*blah):(j*blah)+blah] = np.matrix(np.array(layer[i][:,(j*blah):(j*blah)+blah]) * np.array(layer[i-1][:,j]))
#            layer[i] = np.matrix(np.array(layer[i]) * np.array(layer[i-1])) #Keep this and delete for loop for normal
            layer = notlayer #DELETE THIS
#THIS

            tmpInputLayerLength = layer[i].shape[1]
            #***TODO****
            #Do not do this multiply input's later on only according to inputs which have this            
            inputs = ast.literal_eval(data["inputs"])
            for j in range(0, len(inputs)):
                #which weight number(in array form i.e. weight number - 1) contains array input
                layerIndex = int(inputs[j][-4][2:])-1
                #input number in array form (i.e. input number - 1)
                inputIndex = int(inputs[j][-1][5:])-1
                inputLayer.append(np.transpose(layer[i][:,layerIndex]) * np.transpose(np.matrix(x[:,inputIndex])))
#                print('asdf')
#                print(layer[i][:,layerIndex])
#                print(np.matrix(x[:inputIndex]))
                #input number in array form (i.e. input number - 1)
            
        elif i == 0:
            layer[i] = np.matrix(np.array(layer[i]) * np.array(error) * alpha)

        else:
            layer[i] = np.matrix(np.array(layer[i]) * np.array(layer[i-1]))



    currentBias = layer[:]


    #Multiplies by computeLayer(data, inputs, length, x[k])
    #Cannot do this earlier because it should not be trickled down
    for i in range(0, numLayer):
        if i != numLayer-1:
            layer[i] = np.matrix(np.array(layer[i]) * np.array(l[i-1]))

    #Sum Bias
    for i in range(0, len(currentBias)):
        currentBias[i] = list(np.sum(currentBias[i], axis = 0).tolist()[0]) #Not Sure if [0] works; Might need to fix this


    updatedBias = []
    #Flatten Array
    for i in currentBias:
        if len(i) == 1:
            updatedBias.append(i[0]) #Not Sure if [0] works; Might need to fix
        else:
            for j in i:
                updatedBias.append(j)


    #Flatten Structure

    #Use current state of layer to update bias' and multiply by weights to update weights
    for i in range(0, len(layer)):
        layer[i] = np.transpose(layer[i])


    updatedLayer = []
    other = []
    for i in range(0, len(layer)):
        tmpupdatedLayer = []
        if len(layer[i]) > 1:
            for j in range(0, len(layer[i])):
                updatedLayer.append(np.sum(layer[i][j]))
                tmpupdatedLayer.append(np.sum(layer[i][j]))
        else:
            updatedLayer.append(np.sum(layer[i]))
            tmpupdatedLayer.append(np.sum(layer[i]))
        other.append(tmpupdatedLayer)

    for i in range(0, len(inputLayer)):
       inputLayer[i] = inputLayer[i].tolist()[0][0] #Not sure if 0 always will work; Might need to fix this
    for i in range(0, tmpInputLayerLength):
        updatedLayer.pop(len(updatedLayer)-1)


    for i in range(0, len(inputLayer)):
        updatedLayer.append(inputLayer[i])



    thistmp = []
    thispath = []
#    print("weights")
#    print(weights)
    #Order Layers -- Don't know if this is right; Might need to fix this
    for i in range(0, numLayer):
        for j in range(0, len(repeatWeights)):
            if int(repeatWeights[j][1]) == (numLayer-i):
                thistmp.append(weights[j])
                thispath.append(weightPaths[j])
    weights = np.array(thistmp)
    weightPaths = thispath
    #PRINT SECTION
#    print()
#    for i in updatedLayer:
#        print(i)
#    for i in updatedBias:
#        print(i)
#    print()


    updatedLayer = np.array(updatedLayer)
    updatedLayer = np.array(updatedLayer) + np.array(weights)
    updatedBias = np.array(updatedBias) + np.array(bias)
    #Update Values of Weights in data structure
    for i in range(0, len(updatedLayer)):
        weightPaths[i].append("value")
        setWeights(data, weightPaths[i], updatedLayer[i])
    #Update Value of Bias in data structure
    for i in range(0, len(updatedBias)):
        setWeights(data, biasPaths[i], updatedBias[i])


    return data

def getFromDict(dataDict, mapList):
    return reduce(lambda d, k: d[k], mapList, dataDict)


def setWeights(data, weightPath, value):
    getFromDict(data, weightPath[:-1])[weightPath[-1]] = value

                

#def fixStructure(weights, repeatWeights, weightPaths, layer):
#    group = []
#    for j in range(1, layer+1):
#        tmpgroup = []
#        for i in range(0, len(repeatWeights)):
#            if int(repeatWeights[i][1]) == j:
#                tmpgroup.append(repeatWeights[i])
#        group.append(tmpgroup)
#    print(group)
#        
#    return []



def sigmoid(a):
    a = np.array(a)
    return list(1/(1+np.exp(-a)))

def derSigmoid(a):
    a = np.array(a)
    return list(( np.exp(-1 * a) ) / (np.square(1+np.exp(-1*a))))


"""
def find(data, search, path=[[]]):
    a = []
    for i in data:
        if(i == search):
            a.append(i)
            path[0].append(i)
        if(isinstance(data[i], dict)):
            b = find(data[i], search, path)[0]
            if b:
                for j in b:
                   a.append(j)
                   path[0].append(i)
#                   path.append([])
    return a,path
"""        



#TODO: Check if lastLayerDerSigmoid computes correctly
#Compute Value up until Layer is reached
#Take out path parameter and replace it with path = data["inputs"]; path = ast.literal_eval(path)
def computeLayer(data, path, layer, inputs, lastLayerDerSigmoid=False):
    net=[]
    bias=[]
    sigsum=[]
    inputOrder = []
    for j in range(0,len(path)):
        inputOrder.append(int(path[j][len(path[j])-1][len("input"):])-1)

    #For Each Input Group
    #For Each Input from 0 to the layer parameter
    curInput = 0
    net = inputs
    tmpWeights = []
    for j in range(0,layer+1):
        testNet = []
        for i in range(0,len(path)):
            totalLayers = len(path[i])
            curInput = path[i][len(path[i])-1]
            path[i].pop(len(path[i])-1)
            #If Weight then multiply
            #*** w is reserved for weight ***
            #*** i is reserved for input ***
            #Get Values for Weight
            if curInput[0] == 'w':
                #Go through all weights and see if entire path is same... if so then get rid of redunancies
                path[i].append(curInput)
                tmpWeights.append(path[i][len(path[i])-1])
                path[i].append("value")
                curWeight = getValue(data, path[i])
                testNet.append(curWeight)
                path[i].pop(len(path[i])-1)
                path[i].pop(len(path[i])-1)
            #Get Values for Sigmoid
            elif curInput == "sigmoid":
                path[i].append("sigmoid")
                path[i].append("id")
                sigsum.append(getValue(data, path[i]))
                path[i].pop(len(path[i])-1)
                path[i].append("bias")
                bias.append(getValue(data,path[i][:]))
                path[i].pop(len(path[i])-1)
                path[i].pop(len(path[i])-1)
        #Compute Weight
        if curInput[0] == 'w':
            count =[]
            if len(net) != len(testNet):
                tmp = 0
                for l in inputOrder:
                    testNet[tmp] *= net[l]
                    tmp += 1
                net = testNet
            else:
                for l in range(0,len(testNet)):
                    net[l] *= testNet[l]
                sigsum = []
            tmpWeights = []
        #Compute Sigmoid
        elif curInput == "sigmoid":
            count = []
            for l in range(0,len(sigsum)):
                for k in range(len(sigsum[:l])+1, len(sigsum)):
                    if sigsum[l] == sigsum[k]:
                        net[l] += net[k]
                        #To make sure more than one repeat to take out isn't considered
                        if (k in count) == False:
                            count.append(k)
            #Take out redunancies in reverse order so that the the next one taken out won't be in the wrong order
            for l in sorted(count, reverse=True):
                net = list(net)
                net.pop(l)
                bias.pop(l)
                path.pop(l)
            net = list(np.array(net)+ np.array(bias))
            list(net)
            if lastLayerDerSigmoid == True and j == layer:
                net = derSigmoid(net)
            else:
                net = sigmoid(net)
            bias = []

    return net   





main()

