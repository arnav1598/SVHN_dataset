# The Street View House Numbers Dataset

SVHN is a real-world image dataset for developing machine learning and object recognition algorithm. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.

# About

The model attempts to do both detection using bouding-box regressiong and digit-wise classification of numbers in the dataset using 2 CNNs.

I modified the original SVHN datasset to make it readily workable with keras. The modified dataset contains original train and test images, cropped to bounding box train and test images and two CSV files containing all the information about bounding boxes and labels of train and test images.

The model includes 4 parts:

1. Processing the dataset to make it suitable for working with.
2. Using CNN to do bounding box regression to find the top, left, bottom and right of the bounding box which contains all the digits in a given image.
3. Use the bounding box from step 1 to extract a part of the image with only digits and use other CNNs to classify the digits of the cut image.
4. Using the model on a test image and printing the result.

# Applications

1. A high-end use of the model is to use it for marking house/street numbers in google maps.
2. To evaluate objective mathematics' answer sheets of children studying in primary grades(bounding-box can be already specified).
3. To mark students' attendance on digital platforms using attendance sheets.
