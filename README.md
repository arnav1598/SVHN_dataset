# Project Description:

The Street View House Numbers (SVHN) dataset provides a real-world challenge for developing robust machine learning and object recognition algorithms. Unlike simpler datasets like MNIST, SVHN involves recognizing digits within natural scene images sourced from Google Street View, adding complexity due to varied lighting, occlusions, and diverse backgrounds.

Our project focuses on creating a comprehensive model to tackle this challenge through a two-step approach:

1. Bounding Box Regression: We use a Convolutional Neural Network (CNN) to predict the bounding box coordinates (top, left, bottom, right) that encapsulate all the digits within a given image. This step involves training a CNN to accurately identify these coordinates, effectively isolating regions containing the digits.

2. Digit Classification: Once the bounding boxes are determined, another CNN is employed to classify the digits within these cropped regions. This model is trained to recognize individual digits with high precision, enabling accurate digit identification from the isolated portions of the image.

Dataset Processing: The SVHN dataset is modified and preprocessed to facilitate compatibility with Keras. This includes cropping images to bounding boxes, and organizing associated labels into CSV files for seamless integration into the model.

# Applications:

* Street Address Recognition: Automate the extraction of house numbers from Google Maps for improved address databases and mapping services.
* Educational Tools: Automatically grade primary school math worksheets or other objective answer sheets by detecting and evaluating handwritten digits.
* Attendance Systems: Develop a system for automated marking of attendance sheets, enhancing efficiency and accuracy in educational settings.

This project demonstrates a practical application of advanced computer vision techniques, providing valuable insights into real-world digit recognition and localization challenges.
