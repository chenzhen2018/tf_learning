# -*- coding: utf-8 -*-  
# author: ytq, chenzhen@lognshine.com
# =================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

"""
高级教程-图像分类
https://tensorflow.google.cn/tutorials/images/classification
"""

####################
# Download dataset
####################
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
p_data_dir = Path(data_dir)
print("All image: ", len(list(p_data_dir.glob('*/*.jpg'))))

# show an image
p_roses_dir = list(p_data_dir.glob("roses/*"))
img = Image.open(str(p_roses_dir[0]))
print(img.size, type(img))
img.save("./test/pil_image.png")


#################
# Create Dataset
#################
batch_size = 32
img_height = 180
img_width = 180

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory=str(p_data_dir),
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
ds_val = tf.keras.preprocessing.image_dataset_from_directory(
    directory=str(p_data_dir),
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = ds_train.class_names
print(class_names)

batch_imgs, batch_labels = next(iter(ds_train))
print(batch_imgs, type(batch_imgs))
print(batch_labels, type(batch_labels))

####################
# Visialize the data
####################
plt.figure(figsize=(10, 10))
for images, labels in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig("./test/visualize_the_data.png")
plt.close()

for image_batch, lables_batch in ds_train:
    print(image_batch.shape)
    print(lables_batch.shape)
    break

# Configure the dataset for performace
ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# Standardize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalization_ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
normalization_ds_val = ds_val.map(lambda x, y: (normalization_layer(x), y))

# test normlization
image_batch, labels_batch = next(iter(normalization_ds_train))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

##################
# Build the Model
##################
num_classes = 5
model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
print(model.summary())
# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Model Training
epoches = 10
history = model.fit(ds_train,
                    validation_data=ds_val,
                    epochs=epoches)
############################
# Visualize training results
############################
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epoches), acc, label='Training Accuracy')
plt.plot(range(epoches), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(range(epoches), loss, label='Training Loss')
plt.plot(range(epoches), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title("Training and Vallidation Loss")
plt.savefig('./test/classification.png')
plt.close()

#####################
# Data Augmentation
#####################
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])
# visulaize some images
plt.figure(figsize=(10, 10))
for images, _ in ds_train.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis('off')
plt.savefig("./test/data_augmentataion.png")
plt.close()


################
# Build new Model
#################
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())
epochs = 15
history = model.fit(
    ds_train,
    validation_data=ds_val,
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
plt.savefig('./test/classification_new.png')
plt.close()

##############
# Inference
#############
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.preprocessing.image.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)