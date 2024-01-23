
# Image Captioning In English Language
This project focuses on generating captions for images using the VGG16 model and Encoder Decoder model containing CNN/LSTM. It combines computer vision and natural language processing techniques to generate descriptive captions for images.
## Overview
The goal of this project is to develop a model that can analyze images and generate relevant captions describing the content of the images. The project uses the VGG16 model for image feature extraction and LSTM networks for sequence generation. The model is trained on a dataset consisting of images and their corresponding captions.



## Prerequisites
To run this project, you will need the following:

- Python 3.9
- TensorFlow 2.0
- Keras
- nltk
- PIL (Python Imaging Library)
- matplotlib
## Installation

To run this project, follow the steps below:

```bash
  https://github.com/AliNaveed01/Image-Captioning-Project.git

```
Note: Tensorflow and Keras is required.

for Installation of both, you can write 

## Install TensorFlow
```bash
pip install tensorFlow
pip install keras

```

Incase you have already installed them, you can check their version 
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
```

Then Follow These steps to run the code

- Run the data.ipynb file, it will shorten the data chosen for model by chosing 2K pictures and their related captions, (3 captions for 1 picture),ends up making up a PicCaptions.txt which choses the captions for the taken pictures 
- Now just run the ai-image-captioning-project.ipynb file 





    
## Models Architecture
### Image Feature Extraction:
- The VGG16 model is utilized to extract features from input images. The pre-trained VGG16 model is loaded, and the last fully connected layer is removed to obtain the image feature extraction part of the model. This part takes an image as input and outputs a 4,096-dimensional feature vector.

### Caption Generation:
- The caption generation network takes two inputs: the image features and a partial sequence of words. It predicts the next word in the sequence based on the input image and the partial sequence.

#### Image Feature Processing:
- The image features obtained from the VGG16 model are processed through a dropout layer (to prevent overfitting) and a fully connected layer with 256 units and ReLU activation function.

#### Text Sequence Processing: 
- The input sequence of words is processed through an embedding layer with a vocabulary size of vocab_size and embedding dimension of 256. This layer converts each word in the sequence into a dense vector representation.

#### Sequence Encoding: 
- The embedded sequence is further processed through a dropout layer and an LSTM layer with 256 units. The LSTM layer captures the sequential information and summarizes it into a fixed-length vector representation.
- 
#### Decoder:
- The outputs from the image feature processing and sequence encoding stages are combined using an element-wise addition operation. The result is passed through a fully connected layer with 256 units and ReLU activation.

## Model Architecture:
![Model](https://github.com/AliNaveed01/Image-Captioning-Project/blob/main/output.png)



## Training
The batch size was set to 20, and the number of epochs was 10. During each epoch, the model was trained using a data generator that generated batches of image-caption pairs on-the-fly.

#### To prevent overfitting, the following techniques were used:

•	Dropout layers were added to the model to randomly drop out nodes during training and reduce the risk of over-reliance on specific features.

•	 Early stopping was not used in this particular implementation, but it can be a useful technique to prevent overfitting by stopping training when the model’s performance on a validation set starts to degrade. 

•	The model was saved after each epoch to ensure that the best-performing model was retained, rather than the final epoch’s model, which may have overfit the training data.

•	 The training process involved iterating over the training set for the specified number of epochs, generating batches of image-caption pairs using the data generator, and fitting the model using the fit_generator method.

•	 The number of steps per epoch was calculated based on the size of the training set and the batch size. 

•	The verbose parameter was set to 1 to print the training progress for each epoch


### Evaluation:
•	The BLEU score is calculated using the corpus_bleu() function from the NLTK library. 

•	Two BLEU scores are calculated, one for unigrams only (BLEU-1) and another for unigrams and bigrams (BLEU-2). 

•	The weights parameter is used to specify which n-gram scores to include in the calculation. 

•	For example, weights=(1.0, 0, 0, 0) means only the unigram score is considered, while weights=(0.5, 0.5, 0, 0) means both the unigram and bigram scores are considered, with equal weights.


# Output:
 The final fully connected layer is connected to the output layer with softmax activation. The output layer has a size equal to the vocabulary size (vocab_size) and produces a probability distribution over the words in the vocabulary.

# Training:
 The model is trained using a custom data generator. The generator iterates over the training data in batches, generating pairs of image features and partial  mthe next word in the sequence as the output. The model is trained using the categorical cross-entropy loss function and the Adam optimizer.

# Caption Generation: 
 After training, the model can be used to generate captions for new images. The predict_caption function takes an image and generates a caption by iteratively predicting the next word in the sequence until the end tag is reached.



-------------------------------------------------
# Future Improvements

- Experiment with different architectures and models for better performance.
- Explore advanced techniques such as attention mechanisms and transformer models.
- Increase the size of the training dataset for better generalization.

## Authors

This project was developed by Naveed Ali as part of Artificial Intelligence course at FAST NUCES

- [@AliNaveed01](https://github.com/AliNaveed01)

