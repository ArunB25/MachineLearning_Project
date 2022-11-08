from itertools import product
from turtle import clear
from xmlrpc.client import TRANSPORT_ERROR
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
import torch.nn.functional as F
import os
import pandas as pd


class img_dataset(Dataset):
    def __init__(self,annotations_file = 'Products_formated.csv',images_file = 'Images.csv', img_dir = '/home/arun/Desktop/MachineLearning_Project/formated_images'):
        super().__init__()
        product_labels = pd.read_csv(annotations_file, lineterminator="\n", usecols = ['product_id','category']) #import image id and category
        product_labels['category'] = product_labels['category'].str.split(pat="/").str[0] #split categories to retian highest one
        self.labels_dict = dict(enumerate(product_labels['category'].value_counts().index.tolist())) #create dictionary of categories
        print(self.labels_dict)
        product_labels['category'] = product_labels['category'].replace(list(self.labels_dict.values()),list(self.labels_dict.keys())) # replace categories with integer value7
        self.image_labels = pd.read_csv(images_file, lineterminator="\n", usecols = ['id','product_id']) #import image id and category
        self.image_labels['category'] = [product_labels.loc[product_labels['product_id'] == x, 'category'].iloc[0] for x in self.image_labels['product_id']] #for each image get is product category value
        self.img_dir = img_dir

    def __getitem__(self,idx):
        '''
        idx: image index
        gets item idx in image_labels and returns the image as a a tensor and the corresponding label
        '''
        img_path = os.path.join(self.img_dir, f"{self.image_labels.iloc[idx, 0]}jpg_resized.jpg")
        image = read_image(img_path)
        label = self.image_labels.iloc[idx, 2]
        return image.float(), label

    def __len__(self):
        return len(self.image_labels)


def train(model, epochs = 10):

    optimiser = torch.optim.SGD(model.parameters(),lr=0.001)
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
            torch.nn.Linear(272322,13),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=1)  
        )
    
    def forward(self, X):
        return self.layers(X)


if __name__ == '__main__':
    dataset = img_dataset()
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = CNN()
    train(model)

