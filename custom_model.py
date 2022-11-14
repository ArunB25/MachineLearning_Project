import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch


def train(model, epochs = 10):

    optimiser = torch.optim.SGD(model.parameters(),lr=0.0001)
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for i, (features,labels) in enumerate(train_dataloader):
            prediction = model(features)
            loss = F.cross_entropy(prediction,labels)
            loss.backward()
            print("Loss:",loss.item())
            optimiser.step()
            optimiser.zero_grad()   
            writer.add_scalar('loss',loss.item(),batch_idx)
            batch_idx += 1


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,9,8,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(9,18,8,stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(272322,500),
            torch.nn.ReLU(),
            torch.nn.Linear(500,13),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)  
        )
    
    def forward(self, X):
        return self.layers(X)



if __name__ == '__main__':
    dataset = datasets.img_dataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = CNN()
    train(model)
    torch.save(model,'trained_model#1')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    for i, (features,labels) in enumerate(test_dataloader):
        predictions = model(features)
        print(predictions, labels)
        print(dataset.labels_dict)
        break