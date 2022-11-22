from itertools import product
from turtle import clear
from xmlrpc.client import TRANSPORT_ERROR
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import json


class img_dataset(Dataset):
    def __init__(self,annotations_file = 'Products_formated.csv',images_file = 'Images.csv', img_dir = '/home/arun/Desktop/MachineLearning_Project/formated_images'):
        super().__init__()
        product_labels = pd.read_csv(annotations_file, lineterminator="\n", usecols = ['product_id','category']) #import image id and category
        product_labels['category'] = product_labels['category'].str.split(pat="/").str[0] #split categories to retian highest one
        #self.labels_dict = dict(enumerate(product_labels['category'].value_counts().index.tolist())) #load dictionary of categories
        self.labels_dict = json.load("categories_disct.json")
        product_labels['category'] = product_labels['category'].replace(list(self.labels_dict.values()),list(self.labels_dict.keys())) # replace categories with integer value7
        self.image_labels = pd.read_csv(images_file, lineterminator="\n", usecols = ['id','product_id']) #import image id and category
        self.image_labels['category'] = [product_labels.loc[product_labels['product_id'] == x, 'category'].iloc[0] for x in self.image_labels['product_id']] #for each image get its product category value
        self.img_dir = img_dir

        for file in os.listdir(img_dir):
            file_path = os.path.join(img_dir, file) #Full path to the file
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) #pass the full path to getsize()
                if size == 0:
                    print(file)
                    product_id_no_image = file.split('.', 1)[0]   
                    #self.image_labels['product_id']
                    self.image_labels = self.image_labels[self.image_labels.product_id != product_id_no_image]

    def __getitem__(self,idx,rtn_float = True):
        '''
        idx: image index
        gets item idx in image_labels and returns the image as a a tensor and the corresponding label
        '''
        img_path = os.path.join(self.img_dir, f"{self.image_labels.iloc[idx, 0]}.jpg")
        label = self.image_labels.iloc[idx, 2]
        try:
            image = read_image(img_path)   
        except:
            print("ERROR GETTING ITEMS | image path ", img_path,"Labels",label) 

        if rtn_float:
            return image.float(), label
        else:
            return image, label

    def __len__(self):
        return len(self.image_labels)



if __name__ == '__main__':
    dataset = img_dataset()
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # data_transforms = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   
    train_features, train_labels = next(iter(train_dataloader))
    import matplotlib.pyplot as plt

    for idx, label in enumerate(train_labels):
        img = train_features[idx]
        #plt.subplot(1,2,1)
        plt.imshow(img.permute(1, 2, 0))    
        plt.title(dataset.labels_dict[label.item()])
        # plt.subplot(1,2,2)
        # img = transforms.ToPILImage(img)
        # print(type(img))
        # plt.imshow(data_transforms(img))
        plt.show()
