# Neural Networks - Deep Learning Models in Keras - Classification 
Fully Connected Deep Neural Networks

A fully connected neural network consists of a series of fully connected layers. A fully connected layer is a function from ℝ m to ℝ n . Each output dimension depends on each input dimension. 


The nodes in fully connected networks are commonly referred to as “neurons.” Consequently, elsewhere in the literature, fully connected networks will commonly be referred to as “neural networks.” This nomenclature is largely a historical accident.


Classification example of mnist Fashion dataset.


The Fashion dataset includes:

#60,000 training examples
#10,000 testing examples
#10 classes
#28×28 grayscale/single channel images
#The ten fashion class labels include:
#
#1     T-shirt/top  
#2     Trouser/pants
#3     Pullover shirt
#4     Dress
#5     Coat
#6     Sandal
#7     Shirt
#8     Sneaker
#9     Bag
#10    Ankle boot

from keras.datasets import fashion_mnist:
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

Returns:
2 tuples:
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of labels (integers in range 0-9) with shape (num_samples,).
