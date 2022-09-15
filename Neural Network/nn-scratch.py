# Try RelU and He initialization    https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#---------------------------------------------------------------------------#

import numpy as np
import csv


def Sigmoid(weightedInput):
        return (1 / (1 + np.exp(-weightedInput)))

def CostDerivative(output_value, expected_value):
    return 2*output_value - 2*expected_value
    
def SigmoidDerivative(weightedInput):
    activation = Sigmoid(weightedInput)
    return activation*(1 - activation)

def RelU(weightedInput):
    return np.maximum(weightedInput, 0)

def RelUDerivative(weightedInput):
    return (weightedInput > 0)* 1 

def leakyRelU(weightedInput, alpha = 0.1):
    return np.where(weightedInput > 0, weightedInput, weightedInput * alpha)

def leakyRelUDerivative(weightedInput, alpha = 0.1):
    return np.where(weightedInput > 0, 1, alpha) 


class Accuracy:
    def __init__(self):
        self.correct = 0
        self.counter = 0


    def Accuracy(self,  expectedResult, networkResult):
        self.counter += 1
        if expectedResult == networkResult:
            self.correct += 1

        return self.correct / self.counter * 100


class Layer: 
    def __init__(self, numNodesIn, numNodesOut, activationFunction, activationFunctionDerivative):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.weights = (np.random.uniform(low=0.14, high= 0.019 , size=(numNodesOut, numNodesIn)) * np.sqrt(2 / numNodesIn)) #outputs numbers between 0 and 100
        #self.biases = (np.random.uniform(low= 0., high= 0.1, size=(numNodesOut, 1)))
        #self.weights = (np.zeros(shape=(numNodesOut, numNodesIn))) #outputs numbers between 0 and 100
        #self.weights = (np.random.randn(numNodesOut, numNodesIn) * np.sqrt(2 / numNodesIn))
        self.biases = (np.zeros(shape=(numNodesOut, 1)))
        self.weightsGradients = None    #(out, in)matrix
        self.biasGradients = None    #(out, 1) column vector 
        self.nodeValues = None   #
        self.weightedInputs = None   #(out, 1) column vector
        self.activations = None   #(out, 1) column vector
        self.input = None  #(in, 1) column vector
        self.activationFunction = activationFunction
        self.activationFunctionDerivative = activationFunctionDerivative

    #calculate the output of the Layer
    def CalculateOutputs(self, inputMatrix):
        
        weightedInputsMatrix = self.weights.dot(inputMatrix) + self.biases
        activatedOutput = self.activationFunction(weightedInputsMatrix)

        self.weightedInputs = weightedInputsMatrix
        self.activations = activatedOutput

        return activatedOutput








class NeuralNetwork:
    #should be initialized with layers
    def __init__(self, input,*args):
        self.Layers = args
        self.LayerCount = len(args)
        self.Input = input
        self.Output = None
        self.numOut = args[-1].numNodesOut

    #calculate the output of the network
    def CalculateOutputs(self, input):
        for count in range(self.LayerCount):
            self.Layers[count].input = input
            input = self.Layers[count].CalculateOutputs(input)


        
        self.Output = input
        output = input
        indexOfMaxValue = np.where(output == np.amax(output))[0][0]

        return output, indexOfMaxValue


    #forms the expected result vector
    def ExpectedOutputVectorGenerator(self, expectedValue):
        expectedOutput = np.array([0] * self.numOut).reshape(self.numOut, 1)
        expectedOutput[expectedValue] = 1
        return expectedOutput

    #calculates the cost between real output and expected output
    def CalculateCost(self, expectedResultValue):
        expectedOutputVector = self.ExpectedOutputVectorGenerator(expectedResultValue)
        cost = np.square(expectedOutputVector - self.Output)
        return cost



    def GradientsForOutputLayer(self, expectedOutputVector, learnRate):

        OutputLayer = self.Layers[-1]

        CostWRTActivation = 2 * (self.Output - expectedOutputVector)                    #numOut x 1 of Output Layer
        ActivationWRTWeightedInput = OutputLayer.activationFunctionDerivative(OutputLayer.weightedInputs)      #numOut x 1 of Output Layer   
        WeightedInputWRTWeight = self.Layers[-2].activations.T 

        biasGradients = (CostWRTActivation * ActivationWRTWeightedInput) * learnRate
        weightsGradients = biasGradients.dot(WeightedInputWRTWeight)

        OutputLayer.weightsGradients = weightsGradients
        OutputLayer.biasGradients = biasGradients
        return weightsGradients, biasGradients


    def GradientsForHiddenLayers(self, learnRate):
        for layerIterator in range(self.LayerCount - 1):
            NextLayer = self.Layers[-(layerIterator + 1)]
            CurrentLayer = self.Layers[-(layerIterator + 2)]
            

            OutputWeightsTransposed = NextLayer.weights.T                                                             #numIn x numOut of Output Layer
            OutputBiasGradients = NextLayer.biasGradients                                                             #numOut x 1     of Output Layer
            Activation2WRTWeightedInput2 = CurrentLayer.activationFunctionDerivative(CurrentLayer.weightedInputs)                                #numOut x 1     of Hidden Layer
            
            biasGradients = OutputWeightsTransposed.dot(OutputBiasGradients) * Activation2WRTWeightedInput2 * learnRate #numOut x 1 of Hidden Layer
            weightsGradients = biasGradients.dot(CurrentLayer.input.T)                                                               #numOut x numIn of Hidden Layer

            CurrentLayer.weightsGradients = weightsGradients
            CurrentLayer.biasGradients = biasGradients
        

    def CorrectAllWeights(self):
        for layerIterator in range(self.LayerCount):
            self.Layers[layerIterator].weights -= self.Layers[layerIterator].weightsGradients
            self.Layers[layerIterator].biases -= self.Layers[layerIterator].biasGradients


    
    def PrintProperties(self):
        # print("Count: ", counter,"Output: ", resultOutput, " ||  Expected Output: ", expectedValue, " ||  Cost: ", cost)
        # print("OutputVector: ", MintOutput, " || Expected Output : ", expectedOutputVector)
        # print()
        # print("First Layer Weight Gradients: ", Layer1.weightsGradients, "SUM: ", np.sum(Layer1.weightsGradients))
        # print()
        # print("First Layer Bias Gradients: ", Layer1.biasGradients)
        # print()
        # print("First Layer Activations: ", Layer1.activations)
        # print("First Layer WeightedInputs: ", Layer1.weightedInputs)
        # print("First Layer Weights: ", Layer1.weights)
        # print("Second Layer Weight Gradients: ", Layer2.weightsGradients[1])
        # print("Inputs: ",Inputs)
        # print("Layer1 NodeValues: ",Layer1.nodeValues)
        # print()
        # print("Layer2 NodeValues: ",Layer1.nodeValues)
        # print()
        # print("Layer2 WeightedInput: ", Layer2.weightedInputs)
        return 0


    def BatchSeperator(self, epochs, miniBatchSize):
        return 0 


    def TrainNetwork(self, trainData, epochs, miniBatchSize, learnRate):
        print("Network training with {} samples".format(epochs))
        np.random.shuffle(trainData)
        numberOfBatchs = int(epochs / miniBatchSize)
        donePercent = 0
        counter = 0
        for batchNum in range(numberOfBatchs):
            for turnNumber in range(miniBatchSize):

                input = trainData[counter]
                expectedValue = int(input[0])
                Inputs = np.array([float(element)/255 for element in input[1:]]).reshape(784, 1)
                            
                MintNeuralNetwork.CalculateOutputs(Inputs) #turns once and returns output vector and output value                
                expectedOutputVector = MintNeuralNetwork.ExpectedOutputVectorGenerator(expectedValue)
            
                MintNeuralNetwork.GradientsForOutputLayer(expectedOutputVector, learnRate)
                MintNeuralNetwork.GradientsForHiddenLayers(learnRate)
                MintNeuralNetwork.CorrectAllWeights()   

                if counter % int(epochs / 10) == 0:
                    print("Training Status: %{}0".format(donePercent))
                    donePercent += 1

                counter += 1
                if counter > epochs:
                    break
        
        print()
        print("Training Done!")
        print()
    
    def TestNetwork(self, testData, epochs):
        print("Testing Network with {} samples".format(epochs))
        np.random.shuffle(testData)
        counter = 0
        for input in testData:            
            expectedValue = int(input[0])
            Inputs = np.array([float(element)/255 for element in input[1:]]).reshape(784, 1)
                        
            MintOutput, resultOutput = MintNeuralNetwork.CalculateOutputs(Inputs) #turns once and returns output vector and output value
            #print("OutputVector: ", MintOutput, expectedValue == resultOutput)
            
            accuracyPercent = accuracy.Accuracy(expectedValue, resultOutput)
            counter += 1
            if counter > epochs:
                break
            
        print("Test Done with Accuracy: {}".format(accuracyPercent))

file = open('train.csv')
type(file)
csvreader = csv.reader(file)

header = []
header = next(csvreader)

digitImages = []
for row in csvreader:
    digitImages.append(row)


accuracy = Accuracy()
learnRate = 0.04
trainEpochs = 42000
testEpochs = 1000
miniBatchSize = 100

HiddenLayer1 = Layer(784, 16, leakyRelU, leakyRelUDerivative)
HiddenLayer2 = Layer(16, 16, leakyRelU, leakyRelUDerivative)
OutputLayer = Layer(16, 10, leakyRelU, leakyRelUDerivative)

MintNeuralNetwork = NeuralNetwork(digitImages, HiddenLayer1, HiddenLayer2, OutputLayer)

MintNeuralNetwork.TrainNetwork(digitImages, trainEpochs, miniBatchSize, learnRate)
MintNeuralNetwork.TestNetwork(digitImages, testEpochs)
