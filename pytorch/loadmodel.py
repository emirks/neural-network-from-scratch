from json import load
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader

device = "cuda"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  #declared this function for flattening an image input, nn.Flatten() function returns a function
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ) #nn.Sequential function returns a function

    def forward(self, input):
        input = self.flatten(input)
        output = self.linear_relu_stack(input)
        return output


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y : torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
mintLossFunction = nn.CrossEntropyLoss()

def test_network(dataLoader, neuralNetwork, loss_function):
    size = len(dataLoader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for input, expectedOutput in dataLoader:
            input, expectedOutput = input.to(device), expectedOutput.to(device)
            output = neuralNetwork(input)
            test_loss += loss_function(output, expectedOutput).item()
            correct += (output.argmax(1) == expectedOutput.argmax(1)).type(torch.float).sum().item()
    
    accuracy = correct / size * 100

    print(f"Accuracy: %{accuracy}")


loadedModel = NeuralNetwork().to(device)
loadedModel.load_state_dict(torch.load('savedmodel.pt'))
# loadedModel.eval()

test_network(test_dataloader, loadedModel, mintLossFunction)