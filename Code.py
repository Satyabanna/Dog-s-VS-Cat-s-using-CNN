import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data preprocessing

train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
 'dataset/training_set',
 target_size=(64, 64),
 batch_size=32,
 class_mode='categorical') 
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
 'dataset/test_set',
 target_size=(64, 64),
 batch_size=32,
 class_mode='categorical')

# Building a CNN model

# Step-1: Initialization of CNN model
cnn = tf.keras.models.Sequential()

# Step-2: Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,
 activation='relu', input_shape=[64, 64, 3], padding='same'))

# Step-3: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))

# Step-4: Adding a Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))

# Step-5: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step-6: Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step-7: Output
cnn.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# Compile
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
history = cnn.fit(train_generato, validation_data=validation_generator, epochs=30)

# Make single prediction

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image/255.
result = cnn.predict(test_image)
if np.argmax(result) == 0:
 prediction = 'cat'
else:
 prediction = 'dog'

print(prediction)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()