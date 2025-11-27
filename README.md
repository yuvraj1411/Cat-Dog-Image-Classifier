🐾 Cats and Dogs Image Classifier (CNN)

This code implements a complete Convolutional Neural Network (CNN) for the binary classification of images into cats or dogs. The project uses TensorFlow and Keras to build, train, and evaluate a deep learning model. 
🔗 Colab Notebook Link: https://colab.research.google.com/drive/1PFO1YZzRKGkEzvweeL9aM_80k6-ROC1H?usp=sharing

🚀 Key Features and Workflow

The notebook automates the entire process of image classification:

1. Data Acquisition and Preparation: Downloads and unzips the cats_and_dogs.zip dataset, automatically organizing it into train, validation, and test directories. It calculates the total number of images in each       set.
2. Image Augmentation: Utilizes ImageDataGenerator with various techniques (rotation, shift, shear, zoom, horizontal flip) on the training set to artificially expand the dataset and prevent overfitting. All images     are also scaled down to 1/255.
3. Data Loading: Uses flow_from_directory to efficiently load and preprocess images in batches of 32, targeting an image size of 150x150 pixels.
4. Model Architecture: A sequential CNN model is constructed featuring three blocks of Conv2D (32, 64, 128 filters) followed by MaxPooling2D layers to extract hierarchical features. The output is flattened and         passed through a Dense layer (512 units with ReLU, a Dropout layer (0.5) for regularization, and a final Dense layer with Sigmoid activation for binary classification.
5. Training: The model is compiled with the Adam optimizer and binary_crossentropy loss. It is trained for 25 epochs using the augmented training data and validated against the validation set.
6. Evaluation and Visualization: Training and validation accuracy and loss are plotted over the epochs to visualize convergence and identify overfitting.The model predicts the probability of an image being a dog       (or cat) on the independent test set.A custom function plotImages displays a sample of the test images along with their predicted labels and confidence scores.
7. Final Test: A challenge test is performed to calculate the model's accuracy against a known list of test image labels, confirming if the model passes a set performance threshold >= 63% accuracy).

💻 Dependencies and Usage
- Dependencies: TensorFlow/Keras, NumPy, Matplotlib. All required files are downloaded automatically.
- Usage: The code is designed to be executed cell-by-cell in a Google Colab environment. The entire process, from data download to final accuracy scoring, runs without manual file configuration.
