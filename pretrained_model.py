import datasets as Dataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os

class FB_classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        for param in self.resnet50.parameters(): 
            param.requires_grad = False
        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Sequential( ##Change final layer
            torch.nn.Linear(fc_inputs, 13),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)) 

    def forward(self,X):
        return self.resnet50(X)


def train(model,train_data_loader,val_dataloader,test_dataloader,epochs = 10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #run training on the gpu
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    writer = SummaryWriter()    
    batch_idx = 0
    print(device)
    model.to(device)

    for epoch in range(epochs):
        for i, (features, labels) in enumerate(train_data_loader):
            features = features.to(device)
            labels = labels.to(device)
            model.train() 
            optimizer.zero_grad()# Clean existing gradients
            outputs = model(features)# Forward pass - compute outputs on input data using the model
            loss = F.cross_entropy(outputs, labels)# Compute loss
            loss.backward()# Backpropagate the gradients
            optimizer.step() # Update the parameters
            accuracy = torch.sum(torch.argmax(outputs,dim=1) == labels).item()/len(labels)
            print(f"Epoch:{epoch}, Batch number:{i}, Training: Loss: {loss.item()}, Training Accuracy: {np.mean(accuracy)}")
            writer.add_scalar('loss',loss.item(),batch_idx)
            writer.add_scalar('Accuracy',np.mean(accuracy),batch_idx)
            batch_idx += 1

        print('Evaluating on valiudation set')
        val_loss,val_acc = evaluate(model, val_dataloader)
        writer.add_scalar("Loss/Val", val_loss, batch_idx)
        
        now = datetime.now() #save the model
        dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
        path = f'models/model_{dt_string}/'
        os.makedirs(path, exist_ok = True) 
        torch.save(model.state_dict(),os.path.join(path,f'model_epoch{epoch}.pt'))

    print('Evaluating on test set')# evaluate the final test set performance
    test_loss,test_acc = evaluate(model, test_dataloader)
    model.test_loss = test_loss
    return model  # return trained model


def evaluate(model, dataloader):
    losses = []
    accuracy = []
    for batch in dataloader:
        features, labels = batch
        prediction = model(features)
        loss = F.cross_entropy(prediction, labels)
        losses.append(loss.detach())
        acc = torch.sum(torch.argmax(prediction,dim=1) == labels).item()/len(labels)
        accuracy.append(acc)
    avg_acc = np.mean(accuracy)
    avg_loss = np.mean(losses)
    print('Average Loss',avg_loss, "| Average Accuracy", avg_acc)
    return avg_loss,avg_acc


if __name__ == '__main__':
    dataset = Dataset.img_dataset()
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size,val_size,test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    classifier = FB_classifier()

    train(classifier,train_dataloader,val_dataloader,test_dataloader,epochs=100)
