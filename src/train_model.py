import pandas as pd
import sklearn
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('src/landmarks.csv')
X = df.drop('class', axis=1)
y = df.drop(X, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




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



model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(y_train_one_hot.shape[1], activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)


# Fit the model to the training data
history = model.fit(X_train, y_train_one_hot, epochs=100, batch_size=16, validation_data=(X_test, y_test_one_hot))


# Save the model
model.save('models/my_model.h5')