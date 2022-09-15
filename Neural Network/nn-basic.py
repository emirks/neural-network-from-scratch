import math
import numpy as np
import csv


def Sigmoid(weightedInput):
        return (1 / (1 + math.exp(-weightedInput)))

def CostDerivative(output_value, expected_value):
    return 2*output_value - 2*expected_value
    

def SigmoidDerivative(weightedInput):
    activation = Sigmoid(weightedInput)
    return activation*(1 - activation)



class Accuracy:
    def __init__(self):
        self.correct = 0
        self.counter = 0


    def Accuracy(self,  expectedResult, networkResult):
        self.counter += 1
        if expectedResult == networkResult:
            self.correct += 1
        print("Accuracy: ", self.correct / self.counter * 100)
        



class Layer: 
    def __init__(self, numNodesIn, numNodesOut):
        self.numNodesIn = numNodesIn
        self.numNodesOut = numNodesOut
        self.weights = (np.random.uniform(low= 0.,high= 0.1 , size=(numNodesOut, numNodesIn))) #outputs numbers between 0 and 100
        self.biases = (np.random.uniform(low= 0., high= 0.1, size=(numNodesOut)))
        self.weightsGradients = None
        self.biasGradients = None
        self.nodeValues = []
        self.weightedInputs = []
        self.activations = []


    #calculate the output of the Layer
    def CalculateOutputs(self, inputs):
        layerInputsWeighted = []
        nodeOutput = 0
        for currentLayerIndex in range(self.numNodesOut):
            nodeOutput = self.biases[currentLayerIndex]
            for previousLayerIndex in range(self.numNodesIn):
                nodeOutput += inputs[previousLayerIndex] * self.weights[currentLayerIndex][previousLayerIndex]
            
            layerInputsWeighted.append(nodeOutput)

        layerOutputsNormalized = [Sigmoid(value) for value in layerInputsWeighted]

        self.weightedInputs = layerInputsWeighted
        self.activations = layerOutputsNormalized

        return layerOutputsNormalized








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
            input = self.Layers[count].CalculateOutputs(input)
        self.Output = input
        return input

    #calculates the output and gives the index of the biggest one
    def Result(self):
        output = self.CalculateOutputs(self.Input)
        indexOfMaxValue = output.index(max(output))
        return output, indexOfMaxValue

    #forms the expected result vector
    def ExpectedOutputVectorGenerator(self, expectedValue):
        expectedOutput = [0] * self.numOut
        expectedOutput[expectedValue] = 1
        return expectedOutput

    #calculates the cost between real output and expected output
    def CalculateCost(self, expectedResultValue):
        expectedOutputVector = self.ExpectedOutputVectorGenerator(expectedResultValue)
        cost = 0
        for output, expected in zip(self.Output, expectedOutputVector):
            cost += pow(output - expected, 2)
        return cost


    # def NodeValuesForOutputLayer(self, expectedValueVector):
    #     nodevalues = []
        
    #     for row in range(self.Layers[-1].numNodesOut):
    #         costDerivative = CostDerivative(self.Layers[-1].activations[row], expectedValueVector[row])
    #         sigmoidDerivative = SigmoidDerivative(self.Layers[-1].weightedInputs[row])
    #         nodevalues.append(costDerivative)
        
    #     self.Layers[-1].nodeValues = nodevalues
    #     return nodevalues


    # def NodeValuesForHiddenLayer(self, oldNodeValues):
    #     nodeValues = [0] * self.Layers[-2].numNodesOut
    #     weights2 = self.Layers[-1].weights
        
    #     for layer1Iterate in range(self.Layers[-2].numNodesOut):
    #         for layer2Iterate in range(self.Layers[-1].numNodesOut):
    #             sigmoidDerivative = SigmoidDerivative(self.Layers[-2].weightedInputs[layer1Iterate])
    #             nodeValues[layer1Iterate] += sigmoidDerivative * oldNodeValues[layer2Iterate] * weights2[layer2Iterate][layer1Iterate]
      
    #     self.Layers[-2].nodeValues = nodeValues
    #     return nodeValues


    def GradientsForOutputLayer(self, expectedOutputVector, learnRate):

        OutputLayer = self.Layers[-1]
        weightsGradients = np.zeros([OutputLayer.numNodesOut, OutputLayer.numNodesIn])
        biasGradients = np.zeros([OutputLayer.numNodesOut])

        for row in range(OutputLayer.numNodesOut):
            CostWRTActivation = 2 * (OutputLayer.activations[row] - expectedOutputVector[row])
            ActivationWRTWeightedInput = SigmoidDerivative(OutputLayer.weightedInputs[row])
            biasGradients[row] = CostWRTActivation * ActivationWRTWeightedInput * learnRate
            
            for column in range(OutputLayer.numNodesIn):                
                WeightedInputWRTWeight = self.Layers[-2].activations[column]
                weightsGradients[row][column] = CostWRTActivation * ActivationWRTWeightedInput * WeightedInputWRTWeight * learnRate
                

        OutputLayer.weightsGradients = weightsGradients
        OutputLayer.biasGradients = biasGradients
        return weightsGradients, biasGradients


    def GradientsForHiddenLayer(self, Input, expectedOutputVector, learnRate):

        HiddenLayer = self.Layers[-2]
        OutputLayer = self.Layers[-1] 
        weightsGradients = np.zeros([HiddenLayer.numNodesOut, HiddenLayer.numNodesIn])
        biasGradients = np.zeros(HiddenLayer.numNodesOut)
        locked = False

        for row in range(HiddenLayer.numNodesOut):
            Activation1WRTWeightedInput1 = SigmoidDerivative(HiddenLayer.weightedInputs[row])
            for column in range(HiddenLayer.numNodesIn):                
                WeightedInput1WRTWeight = Input[column]
                for sumComponents in range(OutputLayer.numNodesOut):
                    CostWRTActivation2 = 2 * (OutputLayer.activations[sumComponents] - expectedOutputVector[sumComponents])
                    Activation2WRTWeightedInput2 = SigmoidDerivative(OutputLayer.weightedInputs[sumComponents])
                    WeightedInput2WRTActivation1 = OutputLayer.weights[sumComponents][row]
                    weightsGradients[row][column] += CostWRTActivation2 * Activation2WRTWeightedInput2 * WeightedInput2WRTActivation1 * Activation1WRTWeightedInput1 * WeightedInput1WRTWeight * learnRate
                    
                    
        for row in range(HiddenLayer.numNodesOut):
            Activation1WRTWeightedInput1 = SigmoidDerivative(HiddenLayer.weightedInputs[row])
            WeightedInput1WRTBias = 1.
            for sumComponents in range(OutputLayer.numNodesOut):
                CostWRTActivation2 = 2 * (OutputLayer.activations[sumComponents] - expectedOutputVector[sumComponents])
                Activation2WRTWeightedInput2 = SigmoidDerivative(OutputLayer.weightedInputs[sumComponents])
                WeightedInput2WRTActivation1 = OutputLayer.weights[sumComponents][row]
                biasGradients[row] += CostWRTActivation2 * Activation2WRTWeightedInput2 * WeightedInput2WRTActivation1 * Activation1WRTWeightedInput1 * WeightedInput1WRTBias * learnRate

        HiddenLayer.weightsGradients = weightsGradients
        HiddenLayer.biasGradients = biasGradients

    def CorrectAllWeights(self):
        self.Layers[-2].weights -= self.Layers[-2].weightsGradients
        self.Layers[-2].biases -= self.Layers[-2].biasGradients
        self.Layers[-1].weights -= self.Layers[-1].weightsGradients
        self.Layers[-1].biases -= self.Layers[-1].biasGradients


    


file = open('train.csv')
type(file)
csvreader = csv.reader(file)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

np.random.shuffle(rows)




accuracy = Accuracy()
learnRate = 1
miniBatchSize = 100


Layer1 = Layer(784, 16)
Layer2 = Layer(16, 10)
print("MAX Layer1 Weight: ", max(Layer1.weights[1]))
print("MAX Layer2 Weight: ", max(Layer2.weights[1]))
Layer1.weightedInputs.append(0)
Layer2.weightedInputs.append(0)

maxweighedInputs1, maxweighedInputs2, minweighedInputs1, minweighedInputs2 = [], [], [], []

counter = 0
for input in rows:
    expectedValue = int(input[0])
    Inputs = [float(element)/255 for element in input[1:]]
    MintNeuralNetwork = NeuralNetwork(Inputs, Layer1, Layer2)
    

    maxweighedInputs1.append(max(Layer1.weightedInputs))
    minweighedInputs1.append(min(Layer1.weightedInputs))
    maxweighedInputs2.append(max(Layer2.weightedInputs))
    minweighedInputs2.append(min(Layer2.weightedInputs))

    print("Max Weighted Input of Layer1: ", max(maxweighedInputs1))
    print("Max Weighted Input of Layer2: ", max(maxweighedInputs2))
    print("Min Weighted Input of Layer1: ", min(minweighedInputs1))
    print("Min Weighted Input of Layer2: ", min(minweighedInputs2))

    
    MintOutput, resultOutput = MintNeuralNetwork.Result() #turns once
    expectedOutputVector = MintNeuralNetwork.ExpectedOutputVectorGenerator(expectedValue)
    cost = MintNeuralNetwork.CalculateCost(expectedValue)
   



    weightandbiasgradients2 = MintNeuralNetwork.GradientsForOutputLayer(expectedOutputVector, learnRate)
    weightandbiasgradients1 = MintNeuralNetwork.GradientsForHiddenLayer(Inputs, expectedOutputVector, learnRate)

    cost = MintNeuralNetwork.CalculateCost(expectedValue)
    MintNeuralNetwork.CorrectAllWeights()


    accuracy.Accuracy(expectedValue, resultOutput)
    print("Count: ", counter,"Output: ", resultOutput, " ||  Expected Output: ", expectedValue, " ||  Cost: ", cost)
    # print("OutputVector: ", MintOutput, " || Expected Output : ", expectedOutputVector)
    # # print()
    # print("First Layer Weight Gradients: ", Layer1.weightsGradients, "SUM: ", np.sum(Layer1.weightsGradients))
    # # print()
    print("First Layer Bias Gradients: ", Layer1.biasGradients)
    # # print()
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


    counter += 1
    if counter > 1000:
        break
    




# print("Second version of biases: ", Layer2.biases)
print(cost)

print()
#print(MintOutput, resultOutput)
print("Training Done!")
#print(cost)


#print(Layer3.weights)
#print(len(Layer3.weights[0]))
#print(Layer1.biases)
#print(type(Layer1))
