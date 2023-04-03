import tensorflow as tf


from hypermodel import HyperModel
from data_vizualization import plot_image, plot_distribution, plot_mean


class Model:
    def __init__(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist

        # Retrieve training and testing dataset from the loaded MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        """
        Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
        To do so, divide the values by 255. 
        It's important that the training set and the testing set be preprocessed in the same way.
        """
        x_train, x_test = x_train / 255.0, x_test / 255.0

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_image(self):
        # Check image content by displaying one digit image from an index input
        plot_image(self.x_train, self.y_train)

    def get_distribution(self):
        # Display digit distribution in training and testing datasets
        plot_distribution(self.y_train, self.y_test)

    def get_mean(self):
        # Display digit mean occurence in training and testing datasets
        plot_mean(self.y_train, self.y_test)

    def train(self, max_epochs):
        """
        Instanciate the hypermodel with current training and testing datasets.
        Also pass the limit of epochs that could will run in order to find the optimal number of epoch
        """
        hypermodel = HyperModel(self.x_train, self.y_train, self.x_test, self.y_test, max_epochs)

        # Build the model and get or set the search
        model = hypermodel.build()

        # First trainign test to find the optimal number of epoch
        epochs = hypermodel.get_best_epoch()

        # Train the model
        print('\nStarting training...')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model.fit(self.x_train, self.y_train, validation_split=0.3, callbacks=[stop_early], epochs=epochs)
        print('\nTraining done !')

        # Save the model so he could be infer an unlimited amount of time without training again
        print('\nSaving...')
        model.save("mnist_model.h5")
        print('\nSaving done !')
