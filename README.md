# MachineLearning_Project
Recreating Facebook Marketplace's Recommendation Ranking System

Initially the datasets needed cleaning. The tabular dataset containing all the details from the market palce listings was cleaned, some of the columns data types were changed to suit them, as nearly all were objects. For example price was change from a object data type to a floating point so that arithmetic operatiors could be used if needed, to change the data type the pounds sign needed to be removed. Listings with descriptions and or locations that were empty or didnt contain characters from the alphabet were removed. 
The Image data set was the next to be cleaned, the pillow libaray was used to resize all images with padding to be the same size and to not distort the images. The image channels were all standardised to the RGB format also.