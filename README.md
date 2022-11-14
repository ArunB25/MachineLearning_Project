# MachineLearning_Project
Recreating Facebook Marketplace's Recommendation Ranking System

Initially the datasets needed cleaning. The tabular dataset containing all the details from the market palce listings was cleaned, some of the columns data types were changed to suit them, as nearly all were objects. For example price was change from a object data type to a floating point so that arithmetic operatiors could be used if needed, to change the data type the pounds sign needed to be removed. Listings with descriptions and or locations that were empty or didnt contain characters from the alphabet were removed. 
The Image data set was the next to be cleaned, the pillow libaray was used to resize all images with padding to be the same size and to not distort the images. The image channels were all standardised to the RGB format also.

With the cleaned data a pytorch dataset which inherited the pytorch dataset class, the magic method __getitem__ was altered to take the inputed index, get the image id associated from the index, then read the image from the images folder into a tensor. Both the image tensor and the corresponding label are returned. The dataset is loaded into a dataloader so that the data can be fed into the training loop for the neural network.

A Convolutional Neural Network built on pytorch using the pytorch sequential class where the layers were defined using various classes from the torch.nn class. This task was carried out to understand the the structure and creation of neural networks

![image](https://user-images.githubusercontent.com/111798251/201718931-3234b9d5-da9c-4db7-b0ab-f9864bdc04bc.png)

The training loop was built utilised pytorchs backward method and optimizer modules to find all the gradients and optimise them based on the current batch.

Fine tune a pre-trained neural network

save the weights whilst training and save final model
![image](https://user-images.githubusercontent.com/111798251/201707502-e9aaec78-72d4-4ee4-a0d3-77f49759d635.png)


![image](https://user-images.githubusercontent.com/111798251/201707618-bda4e64f-ecd6-45f8-bb51-4dee7629689c.png)

create image processor script 
