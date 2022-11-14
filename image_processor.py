import pretrained_model as Model
import datasets as Dataset
import torch
import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
    classifier = Model.FB_classifier()
    state_dict = torch.load("model_epoch79.pt")
    classifier.state_dict(state_dict)

    dataset  = Dataset.img_dataset()

    for i in range(10):
        rand_data = random.randint(0, len(dataset)-1)
        features, label = dataset[rand_data]
        features = torch.unsqueeze(features, 0) #format tensor to 4D for input into model

        predicted_label = torch.argmax(classifier(features))

        title = f"True Label {dataset.labels_dict[label]} | Predicted Label {dataset.labels_dict[predicted_label.item()]}"
        features = torch.squeeze(features)
        features = features.type(torch.uint8)
        plt.imshow(features.permute(1, 2, 0))    
        plt.title(title)
        plt.show()

