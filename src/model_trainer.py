import pickle as pkl
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetV2B3
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from data_ingestion import train_test_data

train_data, val_data = train_test_data()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(228, 228, 3)))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
adam = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='binary_crossentropy', optimizer= adam, metrics = ['accuracy'])
history = model.fit(train_data, epochs=10, validation_data = val_data)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training Acc vs Val Acc')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train, Validation'])
plt.show()

class_names = {v:k for k,v in train_data.class_indices.items()}
print('classes',class_names)
data_model = {'class_name': class_names, 'model':model}
pkl.dump(data_model, open('model_cats_dogs.keras', 'wb'))