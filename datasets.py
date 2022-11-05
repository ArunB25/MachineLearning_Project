from itertools import product
from turtle import clear
from xmlrpc.client import TRANSPORT_ERROR
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import pandas as pd
import matplotlib.pyplot as plt

class img_dataset(Dataset):
    def __init__(self,annotations_file = 'Products_formated.csv',images_file = 'Images.csv', img_dir = '/home/arun/Desktop/MachineLearning_Project/formated_images'):
        super().__init__()
        product_labels = pd.read_csv(annotations_file, lineterminator="\n", usecols = ['product_id','category']) #import image id and category
        product_labels['category'] = product_labels['category'].str.split(pat="/").str[0] #split categories to retian highest one
        self.labels_dict = dict(enumerate(product_labels['category'].value_counts().index.tolist())) #create dictionary of categories
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

        return image, label

    def __len__(self):
        return len(self.image_labels)

if __name__ == '__main__':
    dataset = img_dataset()


    # figure = plt.figure(figsize=(8, 8)) #pots figure of 8 random items in dataset 
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(dataset.image_labels), size=(1,)).item()
    #     img, label = dataset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(dataset.labels_dict[label])
    #     plt.axis("off")
    #     plt.imshow(img.permute(1, 2, 0))
    # plt.show()


    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))

    for idx, label in enumerate(train_labels):
        img = train_features[idx]
        plt.imshow(img.permute(1, 2, 0))
        plt.title(dataset.labels_dict[label.item()])
        plt.show()