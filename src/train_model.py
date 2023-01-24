import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ModelTraining:
    def __init__(self):
        """
                   This class shall be used for New training as well as transfer learning of data.

                   Written By: Ritik Dutta
                   Version: 1.0
                   Revisions: None

                   """
        pass

    def train_model(self, n_components, epochs, data_directory='src/rirak_multi_face_landmarks.csv',
                    model_output_directory='models/my_model.h5', input_model=None, pca_model=None):
        """
                   This method takes training parameters and fine-tune old model or train a new model from scratch.

                   Written By: Ritik Dutta
                   Version: 1.0
                   Revisions: None

                   """
        df = pd.read_csv(data_directory)
        X_max = df.drop('class', axis=1)
        y = df.drop(X_max, axis=1)

        if pca_model:
            print("Using saved PCA Model with {} n_components".format(n_components))
            # Load the saved PCA model
            pca = joblib.load(pca_model)
        else:
            print("Creating new PCA Model with {} n_components".format(n_components))
            # Create a PCA object with the desired number of components
            pca = PCA(n_components=n_components)
            # Fit the PCA model to data
            pca.fit(X_max)
            # Save the PCA model
            joblib.dump(pca, 'models/pca_model.joblib')
        # Transform the data
        data_reduced = pca.transform(X_max)
        X = pd.DataFrame(data_reduced)

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2)

        # Create a label encoder object
        le = LabelEncoder()

        # Fit the label encoder to the training labels
        le.fit(y_train)

        # Transform the training labels
        y_train_enc = le.transform(y_train)

        # Transform the test labels
        y_test_enc = le.transform(y_test)

        # Encode the target variable
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # One-hot encode the encoded target variable
        y_train_one_hot = to_categorical(y_train_enc)
        y_test_one_hot = to_categorical(y_test_enc)

        if input_model:
            # Load the pre-trained model
            model = load_model(input_model)
            # Freeze the layers of the pre-trained model
            for layer in model.layers:
                layer.trainable = False
            # Get the number of input features of the last layer
            input_shape = model.layers[-1].output_shape[1]
            # Add a new dense layer for the new class
            model.add(Dense(32, input_shape=(input_shape,), activation='relu'))
            model.add(Dense(y_train_one_hot.shape[1], activation='softmax'))
            # Compile the model
            optimizer = Adam(learning_rate=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], run_eagerly=True)
        else:
            # Create a new model
            model = Sequential()
            model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(16, activation='relu'))
            model.add(BatchNormalization())
            model.add(Flatten())

            # Add a dense layer
            model.add(Dense(32, activation='relu'))

            # Add another dropout layer
            model.add(Dropout(0.5))

            model.add(Dense(y_train_one_hot.shape[1], activation='softmax'))

            # Compile the model
            optimizer = Adam(lr=0.01)

            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'], run_eagerly=True)

        # Fit the model to the training data
        history = model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=16,
                            validation_data=(X_test, y_test_one_hot))

        # Save the model
        model.save(model_output_directory)
