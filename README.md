```markdown
# Implement a SVM to classify images of cats and dogs from Kaggle dataset

This project implements a Support Vector Machine (SVM) using a Convolutional Neural Network (CNN) to classify images of cats and dogs from the Kaggle dataset. The model is built using TensorFlow and Keras and aims to distinguish between images of cats and dogs with high accuracy.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Description

This project focuses on classifying images of cats and dogs using a CNN model implemented in TensorFlow and Keras. The dataset used is the 'Dogs vs. Cats' dataset from Kaggle.

## Dataset

The dataset used in this project is the 'Dogs vs. Cats' dataset, available on Kaggle. It contains 25,000 images of cats and dogs, split evenly between the two classes.

## Installation

To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/projectname.git
    cd projectname
    ```

2. Install the necessary dependencies:
    ```bash
    pip install tensorflow keras matplotlib opencv-python
    ```

3. Download the dataset from Kaggle:
    ```bash
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !kaggle datasets download -d salader/dogs-vs-cats
    ```

4. Extract the dataset:
    ```python
    import zipfile
    zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
    zip_ref.extractall('/content')
    zip_ref.close()
    ```

## Usage

To train and evaluate the model, run the following code:

```python
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
    )

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
    )

def process(image, label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), padding = 'valid', activation = 'relu', input_shape = (256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2, padding ='valid'))

model.add(Conv2D(64, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2, padding ='valid'))

model.add(Conv2D(128, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2, padding ='valid'))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

result = model.fit(train_ds, epochs = 10, validation_data = validation_ds)
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

- Conv2D layers with ReLU activation
- Batch Normalization layers
- MaxPooling2D layers
- Dense layers with Dropout
- Sigmoid output layer for binary classification

## Training

The model is trained for 10 epochs using the Adam optimizer and binary crossentropy loss function.

## Evaluation

To evaluate the model, you can plot the training and validation accuracy and loss:

```python
import matplotlib.pyplot as plt

plt.plot(result.history['accuracy'], color = 'red', label = 'train')
plt.plot(result.history['val_accuracy'], color = 'blue', label = 'validation')
plt.legend()
plt.show()

plt.plot(result.history['loss'], color = 'red', label = 'train')
plt.plot(result.history['val_loss'], color = 'blue', label = 'validation')
plt.legend()
plt.show()
```

## Examples

To test the model on new images, use the following code:

```python
import cv2
test_img = cv2.imread('/content/Dog.jpg')
plt.imshow(test_img)

test_img.shape

test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input)

test_img = cv2.imread('/content/Cat.jfif')
plt.imshow(test_img)

test_img.shape

test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input)
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Mohammad Hadi Dhukka - https://www.linkedin.com/in/mohammad-hadi-dhukka-5453611a9/ - dhukkahadi2001@gmail.com

Project Link: https://github.com/AbHadi03/projectname
