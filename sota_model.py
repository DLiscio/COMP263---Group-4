"""
COMP263 - Group 4: Evaluating Deep Neural Networks using the Histopathologic Cancer Detection dataset
State of the Art Model & Transfer Learning
"""
import pandas as pd
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ensure directories/paths exists
image_dir = 'data/train/'
labels_file = 'data/train_labels.csv'
reduced_image_dir = 'data/reduced_train/'
os.makedirs(reduced_image_dir, exist_ok=True)

current_time = time.strftime("%Y%m%d-%H%M%S")
results_file = f'results/sota/run_logs/Transfer_learning_Results{current_time}.txt'
generated_image_dir = f'results/sota/images/run_{current_time}'
os.makedirs(generated_image_dir, exist_ok=True)

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Check if reduced dataset exists (contains at least 1 .tif)
if not any(fname.endswith('.tif') for fname in os.listdir(reduced_image_dir)):
    print("Reduced dataset not found. Reducing now...")

    # Seperate by class
    tumor_images = labels_df[labels_df['label'] == 1]
    non_tumor_images = labels_df[labels_df['label'] == 0]

    # Take 10,000 samples from each class
    tumor_images = tumor_images.sample(10_000, random_state=66)
    non_tumor_images = non_tumor_images.sample(10_000, random_state=66)

    # Combine and shuffle
    labels_df_reduced = pd.concat([tumor_images, non_tumor_images]).sample(frac=1, random_state=66)

    # Reduce image dataset and save
    for idx, row in labels_df_reduced.iterrows():
        image_path = os.path.join(image_dir, f"{row['id']}.tif")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            save_path = os.path.join(reduced_image_dir, f"{row['id']}.tif")
            cv2.imwrite(save_path, img)

    # Save reduced dataset labels
    labels_df_reduced.to_csv('data/reduced_train_labels.csv', index=False)
    print("Reduction complete and saved")


else:
    print("Reduced dataset already exists")

    # Load reduced labels
    labels_df_reduced = pd.read_csv("data/reduced_train_labels.csv")

# Load the reduced image data
image_data = []
for idx, row in labels_df_reduced.iterrows():
    image_path = os.path.join(reduced_image_dir, f"{row['id']}.tif")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_data.append(img)
image_data = np.array(image_data)

# Explore dataset
print("Image Labels .head(): \n", labels_df_reduced.head())
print("\nClass Distribution: \n", labels_df_reduced['label'].value_counts())
print("\nImage Dataset Size: ", len(labels_df_reduced))
print("\nImage Data Shape: ", image_data[0].shape)

# Normalize pixel values
image_data = (image_data / 127.5) - 1

# Reshape data
image_data = image_data.reshape(-1, 96, 96, 1).astype('float32')

# Visualize 12 images from dataset
fig_one = plt.figure(figsize=(8,8))
for i in range(12):
    plt.subplot(4,3, i+1)
    plt.axis('off')
    plt.title(f"Label: {labels_df_reduced.iloc[i*3]['label']}")
    plt.imshow(image_data[i*3], cmap='gray')
plt.savefig(f"{generated_image_dir}/sample_images.png")
plt.close(fig_one)

# Split into training, testing, and validation
X = image_data
y = labels_df_reduced['label'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=66, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=66, stratify=y_temp)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")


# Convert grayscale to RGB for EfficientNet (expects 3 channels)
X_train = np.repeat(X_train, 3, axis=-1)
X_val = np.repeat(X_val, 3, axis=-1)
X_test = np.repeat(X_test, 3, axis=-1)

# Augment Data
augment_data = tf.keras.Sequential([
    RandomRotation(0.2),
    RandomFlip("horizontal"),
    RandomFlip("vertical"),
    RandomTranslation(0.1, 0.1),  
    RandomContrast(0.2),
    RandomZoom(0.2),
    tf.keras.layers.GaussianNoise(0.1)
])

# Create tf datasets
batch_size = 32

train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(14450).batch(batch_size)
val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Augment trainiing data
train = train.map(lambda x, y: (augment_data(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

# Function to handle model creation
def create_model(transfer_learning=True):
    if transfer_learning:
        # Initialize pretrained model if transfer_learning is true
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96,96,3))
        base_model.trainable = False
    else:
        # Initialize untrained model if transfer_learning is false
        base_model = base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(96,96,3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)  
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  #
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)  
    x = Dropout(0.3)(x)  
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

# Callback for early stopping
callback = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
]

# Training parameters
epochs = 25
initial_lr = 0.001
fine_tuning_lr = 0.0001

# Train model from scratch
print("\nTraining EfficientNet Model from Scratch... ")
model, _ = create_model(transfer_learning=False)
model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model_history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callback)

# Train model with transfer learning
print("\nTraining EfficientNet with Transfer Learning 1st Stage...")
transfer_model, base_model = create_model(transfer_learning=True)
transfer_model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

transfer_model_history = transfer_model.fit(train, validation_data=val, epochs=epochs, callbacks=callback)

base_model.trainable = True
layer_count = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:layer_count]:
    layer.trainable = False
transfer_model.compile(optimizer=Adam(learning_rate=fine_tuning_lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

transfer_model_history_2 = transfer_model.fit(train, validation_data=val, epochs=30, callbacks=callback)

# Evaluate both models
print("Evaluating Models on Test Set...")
model_results = model.evaluate(test, verbose=1)
transfer_model_results = transfer_model.evaluate(test, verbose=1)

# Print test results
print("\nUntrained Model Test Results:")
print(f"Loss: {model_results[0]:.4f}")
print(f"Accuracy: {model_results[1]:.4f}")
print(f"AUC: {model_results[2]:.4f}")

print("\nTransfer Learning Model Test Results:")
print(f"Loss: {transfer_model_results[0]:.4f}")
print(f"Accuracy: {transfer_model_results[1]:.4f}")
print(f"AUC: {transfer_model_results[2]:.4f}")

# Function to plot model history
def plot_history(metric):
    plt.figure(figsize=(12,6))

    # Untrained model
    plt.plot(model_history.history[metric], label='Untrained Model (training)')
    plt.plot(model_history.history[f'val_{metric}'], label='Untrained Model (validation)')

    #Transfer Learning
    transfer_train_metric = transfer_model_history.history[metric] + transfer_model_history_2.history[metric]
    transfer_val_metric = transfer_model_history.history[f'val_{metric}'] + transfer_model_history_2.history[f'val_{metric}']
    plt.plot(transfer_train_metric, label='Transfer Learning Model (training)')
    plt.plot(transfer_val_metric, label='Transfer Learning Model (validation)')

    plt.title(f'Model {metric}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(f'{generated_image_dir}/{metric}_comparison.png')
    plt.close()

# Plot accuracy and loss
plot_history('accuracy')
plot_history('loss')

# Save training history and results to file
with open(results_file, 'w') as f:
    f.write("EfficientNet Model Training Results\n")
    f.write("================================\n\n")
    
    f.write("From Scratch Model Results:\n")
    f.write(f"Loss: {model_results[0]:.4f}\n")
    f.write(f"Accuracy: {model_results[1]:.4f}\n")
    f.write(f"AUC: {model_results[2]:.4f}\n\n")
    
    f.write("Transfer Learning Model Results:\n")
    f.write(f"Loss: {transfer_model_results[0]:.4f}\n")
    f.write(f"Accuracy: {transfer_model_results[1]:.4f}\n")
    f.write(f"AUC: {transfer_model_results[2]:.4f}\n")