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

It's possible that a dataset is not split into training, validation and test data yet. This can be done during training, but if you plan on augmenting the data before training, it might be wise to stay with the original data for validation purposes

	from sklearn.model_selection import train_test_split
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

### One-Hot-Encoding

For classification problems labels are usually represented by integer numbers, where a specific number is arbitrarily assigned to a certain label or class. Neural networks usually have exactly as many nodes in the final layer as there are classes. To facilitate comparing the predicted values with the known labels, so-called one-hot-encoding is used. Labels such as `[0, 1, 2, 2, 1, 2]` are converted into a two-dimensional array, where the respective position is set to `1` and all the others are set to `0`: `[[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1] ...]`.

Keras has a built-in function for this:

	labels_onehot = keras.utils.to_categorical(labels, num_classes=3)

This encoding can also be realized as part of a Tensorflow graph, or using the Sklearn library.

	from sklearn.preprocessing import LabelBinarizer
	lb = LabelBinarizer()
	lb.fit(labels)
	labels_onehot = lb.transform(labels)

Or condensed into one line

	labels_onehot = LabelBinarizer().fit_transform(labels)

You can also use `numpy` to obtain a one-hot encoding:

	labels_onehot = np.zeros((labels.size, labels.max()+1), np.uint8)
	labels_onehot[np.arange(labels.size), labels] = 1

Source: [http://stackoverflow.com/questions/29831489/numpy-1-hot-array](http://stackoverflow.com/questions/29831489/numpy-1-hot-array "Stackoverflow")  

## Sequential model

The Keras sequential model is using exactly one input layer and one output layer, between which data is flowing sequentially from one layer to the next. There also exists a functional model that allows more complex structures, such as cyclical or recurrent neural networks. 

### Initialize the model

Initialization creates an empty model.

	model = keras.models.Sequential()

### Define the input layer

The first or input layer can be any kind of layer but requires that the shape of the input is defined, so Keras knows how to handle it. In the case of a square RGB image with a side length of 32 pixels this would be `(32, 32, 3)`. It is thus sensible to first flatten the shape to a one-dimensional vector.

	model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

If input data already is one-dimensional, simply pass the length of the array as `input_dim`

	model.add(keras.layers.Dense(8, input_dim=8))

### Define other layers

Other common layers are

- fully connected layers: `model.add(keras.layers.Dense(output_dim))` with *`output_dim`* being the number of output nodes of the layer,
- activation functions: `model.add(keras.layers.Activation(activation))` with *`activation`* being the name of the activation function to be used, e. g. `"relu"`, `"softmax"` or `"sigmoid"` (see also [https://keras.io/activations/](https://keras.io/activations/ "Activation functions")), as well as
- convolutional layers: `model.add(keras.layers.convolutional.Convolution2D(nb_filter, nb_rows, nb_cols, border_mode=border_mode, subsample=(s_row, s_col)))` with *`nb_filter`* the number of different filters and *`nb_rows`* and *`nb_cols`* the size of each filter. *`border_mode`* determines the kind of padding used: `"valid"` disables padding, `"same"` pads the input with zeros such that the output shape is the same as the input shape (assuming stride length of 1). *`s_row`* and *`s_col`* are the stride lengths in row and column direction, respectively. For a 2D convolutional layer to work the features must of course still be in two-dimensional shape, so no flattening may be done beforehand.

### Compiling the model

Before training the model the learning process needs to be set up. The two most important parameters are the optimizer and the loss function, which is a measure of how well the model performs. For classification problems you will want to use the categorical cross-entropy loss function

	model.compile(optimizer="adam", loss="categorical_crossentropy")

For regression problems, which only predict one real value, the mean squared error can be a useful loss function

	model.compile(optimizer="adam", loss="mse")

### Training the model

This is where the model is finally trained to determine output from input. If you have already split data into training and validation sets, use

	model.fit(features_train, labels_train, validation_data=(features_valid, labels_valid))

If you didn't, Keras will do it for you

	model.fit(features, labels, validation_split=0.2, shuffle=True)

You can set the number of training epochs and the batch size yourself

	model.fit(features, labels, epochs=10, batch_size=32)

### Using the model

Finally, it's possible to use the compiled and trained model to predict values yourself

	prediction = model.predict(input) 

### Saving and reusing the model

If you want to reuse this model at a later point, or outside the current Python script, you simply

	model.save("filename.h5")

and in the new script load it using

	model = keras.models.load_model("filename.h5")

Further information can be found at [https://keras.io/](https://keras.io/ "Keras documentation") 