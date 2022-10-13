# Colour Detection ML Project

## Description
This project uses a convolutional neural network (CNN) to classify solid colour images, which are stored using an ImageFolder dataset structure. The program itself was written in Python, and I used the PyTorch library to implement the CNN and datasets.

I used a CNN because of its effectiveness in analyzing images. PyTorch was the obvious choice for implementing the CNN, given its wide range of features and resources online. 

The biggest barrier to implementing this project was learning to use PyTorch, as the library was completely new to me when I started this project. Thankfully, PyTorch has extensive documentation available online. I've linked a couple useful resources in the 'Useful resources' section, so check them out if you want a head start in making a CNN of your own.

In the future, I hope to implement a friendly user interface to let people easily upload their own custom datasets and classes. I plan to take advantage of the intuitive and scalable nature of the ImageFolder dataset structure to make using this CNN as easy as possible for casual fans of machine learning.

## How to install and run the project
The project was written in Jupyter Notebook through a conda environment, but any Python compiler should be fine.
Ensure that you have all the necessary libraries installed. If not, check the 'Useful resources' section.
Download the project, including the data. To run the program, simply run 'main.py' and read the printed output.

## How to use the project
The model is trained on the images in 'data/train_dataset', and then tests itself on the images in 'data/test_dataset'.
To use this model, place your training and testing images in their respective folders and run the program.
To change the number of training epochs (i.e. iterations through the training data), simply change the epochs variable near the top of 'main.py'. More epochs leads to higher accuracy, at the cost of longer processing time.

Note: The CNN can be easily adapted to train on classes other than colours. To do so, replace the folders inside train_data and test_data with the desired classes and fill the class folders with your training/testing data. 


## Useful resources
Building the project:

https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

Understanding and constructing the datasets:

https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/

Setting up Python with the requisite libraries:

https://stackoverflow.com/questions/57735701/cant-import-torch-in-jupyter-notebook#:~:text=First%2C%20enter%20anaconda%20prompt%20and,run%20code%20using%20pytorch%20successfully.
https://donaldpinckney.com/books/pytorch/book/ch1-setup/mac.html
https://stackoverflow.com/questions/47726509/unable-to-run-import-scikit-image-in-jupyter-notebook-despite-successful-install
