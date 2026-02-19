# MNIST Handwritten Digit Classification using Artificial Neural Network (ANN)

**Author:** Sandra Praveen C

This project implements a basic Artificial Neural Network (ANN) to classify handwritten digits (0–9) from the MNIST dataset using TensorFlow and Keras.

---

## Project Objective

The objective of this project is to build and train a fully connected neural network to classify grayscale images of handwritten digits.

This is a **multi-class classification problem** with 10 output classes (digits 0–9).

---

## Dataset

![Image](https://user-images.githubusercontent.com/51207580/59979373-bc052480-9604-11e9-85b8-464367fdc891.png)

![Image](https://www.researchgate.net/publication/339316006/figure/fig2/AS%3A859737742651393%401581988859568/Samples-from-MNIST-dataset-The-source-label-of-the-sub-figures-in-column-0-to-9-is-0-to.ppm)

![Image](https://miro.medium.com/v2/resize%3Afit%3A664/1%2Amy2iycmx0GcRBb209QhlkQ.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A-thfpMCiTzjkbzn9n6YKtw.png)

Dataset used: **MNIST**

* 60,000 training images
* 10,000 test images
* Image size: 28 × 28 pixels
* Grayscale images
* Pixel range: 0–255

The dataset was loaded using:

```python
from tensorflow.keras.datasets import mnist
```

---

## Data Preprocessing

1. Loaded training and testing data.
2. Normalized pixel values to range [0, 1]:

```python
X_train = X_train / 255
X_test = X_test / 255
```

Normalization improves training stability and convergence speed.

---

## Model Architecture – Model 1

```
Input Layer (28 × 28)
↓
Flatten Layer
↓
Dense Layer (128 neurons, ReLU)
↓
Output Layer (10 neurons, Softmax)
```

### Explanation

* **Flatten**: Converts 28×28 image into 784-dimensional vector
* **ReLU activation**: Adds non-linearity
* **Softmax activation**: Outputs probability distribution over 10 classes

Loss Function:

```
sparse_categorical_crossentropy
```

Optimizer:

```
Adam
```

---

## Model Architecture – Model 2 (Deeper Network)

```
Input Layer (28 × 28)
↓
Flatten
↓
Dense (128 neurons, ReLU)
↓
Dense (32 neurons, ReLU)
↓
Output Layer (10 neurons, Softmax)
```

Changes made:

* Added one additional hidden layer (32 neurons)
* Increased number of epochs

This allowed comparison between shallow and deeper architectures.

---

## Training Configuration

* Validation split: 20%
* Model 1: 10 epochs
* Model 2: 25 epochs
* Metrics monitored:

  * Training loss
  * Validation loss
  * Training accuracy
  * Validation accuracy

Training curves were plotted to analyze learning behavior and detect overfitting.

---

## Evaluation

Predictions were generated using:

```python
model.predict(X_test)
```

Final predicted labels were obtained using:

```python
y_pred = y_prob.argmax(axis=1)
```

Model accuracy was computed using:

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

## Results

* Successfully classified handwritten digits.
* Observed improved performance with deeper architecture.
* Learned how architecture depth affects model performance.

---

## Key Concepts Learned

* Artificial Neural Networks (ANN)
* Forward propagation
* Backpropagation
* Multi-class classification
* Softmax activation
* Sparse categorical crossentropy
* Model evaluation using accuracy
* Training vs Validation performance
* Overfitting vs Underfitting

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## Future Improvements

* Add Dropout layers
* Add L2 regularization
* Implement Early Stopping
* Compare with Convolutional Neural Network (CNN)
* Hyperparameter tuning
* Confusion matrix visualization

---

This project is part of my deep learning learning journey and serves as a foundational implementation of Artificial Neural Networks for image classification.

