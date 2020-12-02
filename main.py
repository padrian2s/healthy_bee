import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import PIL
import tensorflow as tf
import pathlib

from PIL import ImageShow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

INPUT_PATH = "/Users/p3700676/api/"

my_data = pd.read_csv(os.path.join(INPUT_PATH, "bee_data.csv"))
dataset_path = os.path.join(INPUT_PATH, "bee_imgs")

data_dir = pathlib.Path(dataset_path)

healthy = list(data_dir.glob('healthy/*'))
try:
    PIL.Image.open(str(healthy[0]))
except:
    print("there are no healthy bees")
    pass

#Use to print out how many photos are in all the photos in the bee_imgs directory
#image_count = len(list(data_dir.glob('*/*.png')))
#print(image_count)

#use to print out how many photos per category in the CSV
print(my_data['health'].value_counts())

# Create Dataset
# Validation split splits the images into 80% for training and 20% for validation

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
#Prints how many classes are in the new dataset
#print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#         plt.show() #NEEDED. Will not display images unless this is added.

# Configure Dataset
# dataset.cache(): keeps images in memory after they're loaded off the disk for the first epoch
# dataset.prefetch(): overlaps data preprocessing and model execution while training

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create Model
# Consists of three convolutional blocks with each having a max pooling layers
# The fully connected layer is activated by the ReLU function
#
# ​

num_classes = 6
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# Compile Data
# Use Adam optimizer and the Sparse Categorical Croessentropy loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model summary

model.summary()


# Train Model
epochs=1
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss.png')

# prediction
bee_img = "/Users/p3700676/validation/046_193.png"
img = keras.preprocessing.image.load_img(
    bee_img, target_size=(img_height, img_width)
)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to the category {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)



