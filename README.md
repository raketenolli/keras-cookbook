# Keras Cookbook

## Required libraries

### Tensorflow

`import tensorflow as tf`

### Numpy

`import numpy as np`

### Pickle

`import pickle`

### Keras

`import keras`

## Loading and saving data

A file is opened using `with open(filename) as f` instead of `f = open()` to handle errors more gently. First, all the data is loaded into one variable from a pickle file. Usually, data has been saved as a dictionary so separate variables can be extracted from it.

	filename = "data.p"  
	with open(filename, "rb") as f:  
    	data = pickle.load(f)  
	features = data["features"]  
	labels = data["labels"]

Similarly, data is saved into pickle files.

	with open(filename, "wb") as f:
		pickle.dump({"features": features, "labels": labels}, f)

## Preprocessing data

### Reshaping or flattening data

### Resizing

### Normalization

Depending on the features and the machine learning algorithms used it is encouraged (and can never hurt) to normalize data, i. e. the mean is moved to 0 and the values lie within a certain range, in this case [-1, 1].

	fmin = np.amin(features)
	fmax = np.amax(features)
	features_norm = -1.0 + 2.0 * (features - fmin) / (fmax - fmin)

In some cases the interval for the original values is already known, e. g. for RGB-colored images. In this case `fmin` and `fmax` can be assigned constant values. It is also possible that different features need to be scaled differently, e. g. if they represent different physical quantities (pressure, temperature, length, velocity, ...).

### Shuffling data

Shuffling data is useful to allow cross validation and to not always have the same samples in training batches. There is a ready-made function making sure that the correct `labels` are still assigned to the respective `features`.

	from sklearn.utils import shuffle
	features, labels = shuffle(features, labels)

### Splitting data

It's possible that a dataset is not split into training, validation and test data yet.

	from sklearn.model_selection import train_test_split
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

### One-Hot-Encoding

For classification problems labels are usually represented by integer numbers, where a specific number is arbitrarily assigned to a certain label or class. Neural networks usually have exactly as many nodes in the final layer as there are classes. To facilitate comparing the predicted values with the known labels, so-called one-hot-encoding is used. Labels such as `[0, 1, 2, 2, 1, 2]` are converted into a two-dimensional array, where the respective position is set to `1` and all the others are set to `0`: `[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1] ...]`.

This encoding can be realized as part of a Tensorflow graph, but since we are focussing on Keras here, we use the Sklearn library.

	from sklearn.preprocessing import LabelBinarizer
	lb = LabelBinarizer()
	lb.fit(labels)
	labels_onehot = lb.transform(labels)

Or condensed into one line

	labels_onehot = LabelBinarizer().fit_transform(labels)

## Sequential model

The Keras sequential model is using exactly one input layer and one output layer, between which data is flowing sequentially from one layer to the next. There also exists a functional model that allows more complex structures, such as cyclical or recurrent neural networks. 

### Initialize the model

Initialization creates an empty model.

	model = keras.models.Sequential()

### Define the input layer

The first or input layer can be any kind of layer but requires that the shape of the input is defined, so Keras knows how to handle it. In the case of a square RGB image with a side length of 32 pixels this would be `(32, 32, 3)`. It is thus sensible to first flatten the shape to a one-dimensional vector.

	model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

### Define other layers

Other common layers are

- fully connected layers: `model.add(keras.layers.Dense(output_dim))` with *`output_dim`* being the number of output nodes of the layer,
- activation functions: `model.add(keras.layers.Activation(activation))` with *`activation`* being the name of the activation function to be used, e. g. `"relu"`, `"softmax"` or `"sigmoid"` (see also [https://keras.io/activations/](https://keras.io/activations/ "Activation functions")), as well as
- convolutional layers: `model.add(keras.layers.convolutional.Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=border_mode, subsample=(s_row, s_col)))` with *`nb_filter`* the number of different filters and *`nb_rows`* and *`nb_cols`* the size of each filter. *`border_mode`* determines the kind of padding used: `"valid"` disables padding, `"same"` pads the input with zeros such that the output shape is the same as the input shape (assuming stride length of 1). *`s_row`* and *`s_col`* are the stride lengths in row and column direction, respectively. For a 2D convolutional layer to work the features must of course still be in two-dimensional shape, so no flattening may be done beforehand.

Further information can be found at [https://keras.io/](https://keras.io/ "Keras documentation") 