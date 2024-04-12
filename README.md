# Dog Breed Identifier

## Introduction
This Dog Breed Identifier is a machine learning model built with TensorFlow and integrated with Gradio, a library for creating customizable input interfaces. The model is capable of identifying the breed of a dog from an image.

## Dataset
The model is trained on the Stanford Dogs Dataset, which consists of images of 120 dog breeds. I downloaded the dataset from [here](https://www.kaggle.com/competitions/dog-breed-identification/data). Each image in the dataset is labeled with its corresponding breed.

## Model Architecture
The neural network architecture used for this Dog Breed Identifier consists of Xception and InceptionResNetV2
- **Xception**: A deep CNN architecture known for its performance on image classification tasks.
- **InceptionResNetV2**: A hybrid CNN architecture combining the Inception and ResNet modules for improved accuracy.

## Custom Input
The Gradio interface allows for the following:
- **Upload Image**: Users can upload an image file of a dog from their device.

## Example Outputs with Gradio
![Golden Retriever](https://github.com/ThisIsNotJustin/DogBreed_Identifier/blob/main/examples/goldenretriever.png)
![King Charles Spaniel](https://github.com/ThisIsNotJustin/DogBreed_Identifier/blob/main/examples/kingcharlesspaniel.png)
![Samoyed](https://github.com/ThisIsNotJustin/DogBreed_Identifier/blob/main/examples/samoyed.png)
