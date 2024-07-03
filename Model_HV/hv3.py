import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMAGE_SIZE = 256
BATCH_SIZE = 32
Channels = 3
EPOCHS = 50

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset_256x256",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(class_names)
print(f"Number of batches in dataset: {len(dataset)}")

# Display some images from the dataset
plt.figure(figsize=(7, 7))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
    plt.show()

# Function to split the dataset into training, validation, and test sets
def get_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = tf.data.experimental.cardinality(ds).numpy()
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_test_ds = ds.skip(train_size)
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)
    
    return train_ds, val_ds, test_ds

# Split the dataset
train_ds, val_ds, test_ds = get_partitions_tf(dataset)

# Print the sizes of the datasets
print(f"Training set size: {tf.data.experimental.cardinality(train_ds).numpy()} batches")
print(f"Validation set size: {tf.data.experimental.cardinality(val_ds).numpy()} batches")
print(f"Test set size: {tf.data.experimental.cardinality(test_ds).numpy()} batches")

# Optimize dataset performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Normalization layer
normalization_layer = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

# Example of using the data augmentation layer
for image_batch, _ in train_ds.take(1):
    first_image = image_batch[0]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy())
        plt.axis("off")
    plt.show()

# Define the input shape
input_shape = (IMAGE_SIZE, IMAGE_SIZE, Channels)

# Number of classes (assuming binary classification for hallux valgus and normal)
n_classes = len(class_names)

# Define the custom model
custom_model = models.Sequential([
    normalization_layer,
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

# Build the custom model
custom_model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, Channels))
custom_model.summary()

# Define the ResNet50 model
resnet_model = models.Sequential([
    normalization_layer,
    data_augmentation,
    ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, Channels), include_top=False, weights='imagenet'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

# Build the ResNet50 model
resnet_model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, Channels))
resnet_model.summary()

# Compile the models
custom_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

resnet_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the custom model
custom_history = custom_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# Train the ResNet50 model
resnet_history = resnet_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# Evaluate the custom model
custom_scores = custom_model.evaluate(test_ds)
print(f"Custom model test accuracy: {custom_scores[1]*100:.2f}%")

# Evaluate the ResNet50 model
resnet_scores = resnet_model.evaluate(test_ds)
print(f"ResNet50 model test accuracy: {resnet_scores[1]*100:.2f}%")

# Plot the training history for the custom model
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), custom_history.history['accuracy'], label="Training Accuracy")
plt.plot(range(EPOCHS), custom_history.history['val_accuracy'], label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Custom Model Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), custom_history.history['loss'], label="Training Loss")
plt.plot(range(EPOCHS), custom_history.history['val_loss'], label="Validation Loss")
plt.legend(loc='upper right')
plt.title('Custom Model Training and Validation Loss')
plt.show()

# Plot the training history for the ResNet50 model
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), resnet_history.history['accuracy'], label="Training Accuracy")
plt.plot(range(EPOCHS), resnet_history.history['val_accuracy'], label="Validation Accuracy")
plt.legend(loc='lower right')
plt.title('ResNet50 Model Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), resnet_history.history['loss'], label="Training Loss")
plt.plot(range(EPOCHS), resnet_history.history['val_loss'], label="Validation Loss")
plt.legend(loc='upper right')
plt.title('ResNet50 Model Training and Validation Loss')
plt.show()

# Confusion Matrix for the custom model
y_true = []
y_pred = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    predictions = custom_model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Custom Model')
plt.show()

# Confusion Matrix for the ResNet50 model
y_true = []
y_pred = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    predictions = resnet_model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for ResNet50 Model')
plt.show()

# Make predictions on the test set for the custom model
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    prediction_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return prediction_class, confidence

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(custom_model, images[i])
        actual_class = class_names[labels[i].numpy()]
        plt.title(f"Actual: {actual_class}, \nPredicted: {predicted_class} ({confidence}%)")
        plt.axis("off")
plt.show()

# Make predictions on the test set for the ResNet50 model
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(resnet_model, images[i])
        actual_class = class_names[labels[i].numpy()]
        plt.title(f"Actual: {actual_class}, \nPredicted: {predicted_class} ({confidence}%)")
        plt.axis("off")
plt.show()
