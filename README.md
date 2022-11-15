# MachineLearning_Project
Recreating Facebook Marketplace's Recommendation Ranking System

##Cleaning data
Initially the datasets needed cleaning. The tabular dataset containing all the details from the market palce listings was cleaned, some of the columns data types were changed to suit them, as nearly all were objects. For example price was change from a object data type to a floating point so that arithmetic operatiors could be used if needed, to change the data type the pounds sign needed to be removed. Listings with descriptions and or locations that were empty or didnt contain characters from the alphabet were removed. 
The Image data set was the next to be cleaned, the pillow libaray was used to resize all images with padding to be the same size and to not distort the images. The image channels were all standardised to the RGB format also.

## Dataset and Dataloaders
With the cleaned data a pytorch dataset which inherited the pytorch dataset class, the magic method __getitem__ was altered to take the inputed index, get the image id associated from the index, then read the image from the images folder into a tensor. Both the image tensor and the corresponding label are returned. The dataset is loaded into a dataloader so that the data can be fed into the training loop for the neural network.

## Building a Convolutional Neural Network
A Convolutional Neural Network built on pytorch using the pytorch sequential class where the layers were defined using various classes from the torch.nn class. This task was carried out to understand the the structure and creation of neural networks

![image](https://user-images.githubusercontent.com/111798251/201718931-3234b9d5-da9c-4db7-b0ab-f9864bdc04bc.png)

The training loop was built utilised pytorchs backward method and optimizer modules to find all the gradients and optimise them based on the current batch.

## Fine Tuning and Pre-trained neural networks
As creating a custom machine learning model requires an abundance of time and energy spent on defining and optimizing the layers, capacity, training the model with a large and varied dataset and tuning the hyper parameters. A pre-trained neural network can drastically ease the process of creating a neural network for a problem. They are easier to use as they give you the architecture for free, additionally they typically have better results and typically require need less training. 

Pytorchs resnet50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories. The final layer of this model was tweaked, to output the 13 categories of the facebook marketplace listings.

![image](https://user-images.githubusercontent.com/111798251/201910878-a0e77fd7-bcea-4161-8c79-ccd378811a92.png)

## Training the Resnet50 model for our dataset
The dataset was split up into Training, validation and Test sets to monitor the traing and look out for unwanted errors e.g overfitting.
Tensorboard was used to monitor the loss and accuracy over the training.

Below is the loss for the training set, as cross entropy is used the loss value isnt a gaurnteed indicator for the accuracy of the model. The trend shown is typical of what would be expected close to expontential loss.
![image](https://user-images.githubusercontent.com/111798251/201707502-e9aaec78-72d4-4ee4-a0d3-77f49759d635.png)

The training accuracy curve below shows a trend almost like a mirror image of the loss, with both flattening off around the same batches

![image](https://user-images.githubusercontent.com/111798251/201707618-bda4e64f-ecd6-45f8-bb51-4dee7629689c.png)

To ensure the model isnt overfitting the validation loss was also plotted and displayed a trend of exponetial decay, which assured the model wasnt over fitting.

![image](https://user-images.githubusercontent.com/111798251/201907879-11375486-6cb8-443d-b478-e6e78c151d03.png)
