
#Todo encapsulate printLayer (unsafe function) in a wrapper function and cleanup/refactor code


array = ['sigmoid', 'w']

inputs = 5*5*3
input = "input"
perinput = 1  #Inputs allowed in each weight

layers = 5
layer= [1, 5, 1, 5*3] #Tells how many of something there are at each layer; check if it is equal to layers-1 in length

filename = 'configuration.json'




def main():
    genNeuralNetwork(filename, layer, layers, perinput)

def genNeuralNetwork(filename, layer, layers, perinput):
    f = open(filename, 'w')
    f.write("{\n")
    weightLen = numWeights(layers)
    a, b, c, d, e, g = printlayer(0, layer, layers, f, perinput, weightLen, myInputFunction, cur_weight=[0]*layers)
    paths = c
    f.write(',\n')
    f.write('"inputs": ' + '"' + str(paths) + '"' + ',\n')
    f.write('"input": ' + str(inputs) + ",\n")
    f.write('"layers": ' + str(layers) + '\n')
    f.write("}\n")
    f.close()

def myInputFunction(input, cur_weight, overlap, inputs):
    val = input
    if cur_weight != 1:
        val -= cur_weight-1
        val -= (overlap-1)*(cur_weight-1)
    val = val % inputs
    if val == 0:
        val = inputs
    return val

def numWeights(layers):
    length = 0
    for i in range(0, layers):
        if i%2:
            length+=1
    return length


#Has to be open before writing to file
#To have a Convolational Neural Network (i.e. each input group has part overlap with previous just make pergroup smaller than inputs
def printlayer(i, layer, layers, file, perinput, weightLen, inputFunction, cur_weight=[0], cur_id=0, paths=[], curpath = [], curinput=0, currentLayer=1, overlap=0):
    tabspace = "    "*(i+1)
    if i == 0:
        tabspace = "    "
    nl = '\n'
    #If it is an input layer
    if i == (layers-1):
        for j in range(0, perinput):
            name = '"input' + str(inputFunction(curinput+1, cur_weight[i-1], overlap, inputs)) + '"'
            curpath.append('input' + str(inputFunction(curinput+1, cur_weight[i-1], overlap, inputs)))
            paths.append(curpath[:])
            curpath.pop()
            if j < perinput-1:
                file.write(tabspace + name + ': 0' + ',' + nl)
            else:
                file.write(tabspace + name + ': 0' + nl)
            curinput += 1
#            if curinput == inputs:
#                curinput = 0
        return cur_weight, cur_id, paths, curpath, curinput, currentLayer
    #If it is a weight layer
    elif i % 2:
        for j in range(0, layer[i]):
            curLay = (i-1)/2
            name = '"w' + str(int(weightLen-curLay)) + str(int(cur_weight[i]+1)) + '"'
            curpath.append('w' + str(int(weightLen-curLay)) + str(int(cur_weight[i]+1)))
            file.write(tabspace + name + ": ")
            file.write("{" + nl)
            file.write(tabspace + '    ' + '"value": 0,' + nl)
            cur_weight[i] += 1
            cur_weight, cur_id, paths, curpath, curinput, currentLayer = printlayer(i+1, layer, layers, file, perinput, weightLen, inputFunction, cur_weight, cur_id, paths, curpath, curinput, currentLayer, overlap) #Have to send cur_weight so it doesn't reset to zero
            curpath.pop()
            if j == layer[i]-1:
                file.write(tabspace + '}' + nl) # Don't add comma if end of list
            else:
                file.write(tabspace + '},' + nl) # Add comma if not at end of list
        currentLayer+=1

    #If it is a sigmoid layer
    else:
        for j in range(0, layer[i]):
            name = '"sigmoid"'
            curpath.append('sigmoid')
            file.write(tabspace + name + ': ')
            file.write('{' + nl)
            file.write(tabspace + '    ' + '"id": ' + str(cur_id) + "," + nl)
            file.write(tabspace + '    ' + '"bias": 0,' + nl)
            cur_weight, cur_id, paths, curpath, curinput, currentLayer = printlayer(i+1, layer, layers, file, perinput, weightLen, inputFunction, cur_weight, cur_id, paths, curpath, curinput, currentLayer, overlap) #Have to send cur_weight so it doesn't reset to zero
            curpath.pop()
            cur_id += 1
            if j == layer[i]-1:
                file.write(tabspace + '}' + nl) # Don't add comma if end of list
            else:
                file.write(tabspace + '},' + nl) # Add comma if not at end of list
    return cur_weight, cur_id, paths, curpath, curinput, currentLayer

main()






"""
#Todo encapsulate printLayer (unsafe function) in a wrapper function and cleanup/refactor code


array = ['sigmoid', 'w']

inputs = 9
input = "input"
perinput = 1  #Inputs allowed in each weight

layers = 5
layer= [1, 3, 1, 9] #Tells how many of something there are at each layer; check if it is equal to layers-1 in length

filename = 'configuration.json'




def main():
    genNeuralNetwork(filename, layer, layers, perinput)

def genNeuralNetwork(filename, layer, layers, perinput):
    f = open(filename, 'w')
    f.write("{\n")
    weightLen = numWeights(layers)
    a, b, c, d, e, g = printlayer(0, layer, layers, f, perinput, weightLen, myInputFunction, cur_weight=[0]*layers)
    paths = c
    f.write(',\n')
    f.write('"inputs": ' + '"' + str(paths) + '"' + ',\n')
    f.write('"input": ' + str(inputs) + ",\n")
    f.write('"layers": ' + str(layers) + '\n')
    f.write("}\n")
    f.close()

def myInputFunction(input, cur_weight, overlap, inputs):
    val = input
    if cur_weight != 1:
        val -= cur_weight-1
        val -= (overlap-1)*(cur_weight-1)
    val = val % inputs
    if val == 0:
        val = inputs
    return val

def numWeights(layers):
    length = 0
    for i in range(0, layers):
        if i%2:
            length+=1
    return length


#Has to be open before writing to file
#To have a Convolational Neural Network (i.e. each input group has part overlap with previous just make pergroup smaller than inputs
def printlayer(i, layer, layers, file, perinput, weightLen, inputFunction, cur_weight=[0], cur_id=0, paths=[], curpath = [], curinput=0, currentLayer=1, overlap=0):
    tabspace = "    "*(i+1)
    if i == 0:
        tabspace = "    "
    nl = '\n'
    #If it is an input layer
    if i == (layers-1):
        for j in range(0, perinput):
            name = '"input' + str(inputFunction(curinput+1, cur_weight[i-1], overlap, inputs)) + '"'
            curpath.append('input' + str(inputFunction(curinput+1, cur_weight[i-1], overlap, inputs)))
            paths.append(curpath[:])
            curpath.pop()
            if j < perinput-1:
                file.write(tabspace + name + ': 0' + ',' + nl)
            else:
                file.write(tabspace + name + ': 0' + nl)
            curinput += 1
#            if curinput == inputs:
#                curinput = 0
        return cur_weight, cur_id, paths, curpath, curinput, currentLayer
    #If it is a weight layer
    elif i % 2:
        for j in range(0, layer[i]):
            curLay = (i-1)/2
            name = '"w' + str(weightLen-curLay) + str(cur_weight[i]+1) + '"'
            curpath.append('w' + str(weightLen-curLay) + str(cur_weight[i]+1))
            file.write(tabspace + name + ": ")
            file.write("{" + nl)
            file.write(tabspace + '    ' + '"value": 0,' + nl)
            cur_weight[i] += 1
            cur_weight, cur_id, paths, curpath, curinput, currentLayer = printlayer(i+1, layer, layers, file, perinput, weightLen, inputFunction, cur_weight, cur_id, paths, curpath, curinput, currentLayer, overlap) #Have to send cur_weight so it doesn't reset to zero
            curpath.pop()
            if j == layer[i]-1:
                file.write(tabspace + '}' + nl) # Don't add comma if end of list
            else:
                file.write(tabspace + '},' + nl) # Add comma if not at end of list
        currentLayer+=1

    #If it is a sigmoid layer
    else:
        for j in range(0, layer[i]):
            name = '"sigmoid"'
            curpath.append('sigmoid')
            file.write(tabspace + name + ': ')
            file.write('{' + nl)
            file.write(tabspace + '    ' + '"id": ' + str(cur_id) + "," + nl)
            file.write(tabspace + '    ' + '"bias": 0,' + nl)
            cur_weight, cur_id, paths, curpath, curinput, currentLayer = printlayer(i+1, layer, layers, file, perinput, weightLen, inputFunction, cur_weight, cur_id, paths, curpath, curinput, currentLayer, overlap) #Have to send cur_weight so it doesn't reset to zero
            curpath.pop()
            cur_id += 1
            if j == layer[i]-1:
                file.write(tabspace + '}' + nl) # Don't add comma if end of list
            else:
                file.write(tabspace + '},' + nl) # Add comma if not at end of list
    return cur_weight, cur_id, paths, curpath, curinput, currentLayer

main()
"""
