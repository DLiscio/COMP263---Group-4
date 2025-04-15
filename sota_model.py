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
from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU found. Using CPU with optimized settings...")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

# Preprocessing function
def preprocess_image(img):
    # Resize to 128x128 for MobileNetV2
    img = cv2.resize(img, (128, 128))
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    return img

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
def load_images_batch(image_paths, batch_size=1000):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = preprocess_image(img)
                batch_images.append(img)
        yield np.array(batch_images)

# Load and process images
image_paths = [os.path.join(reduced_image_dir, f"{row['id']}.tif") 
               for _, row in labels_df_reduced.iterrows()]
image_data = []
for batch in load_images_batch(image_paths):
    image_data.append(batch)
image_data = np.concatenate(image_data)

# Explore dataset
print("Image Labels .head(): \n", labels_df_reduced.head())
print("\nClass Distribution: \n", labels_df_reduced['label'].value_counts())
print("\nImage Dataset Size: ", len(labels_df_reduced))
print("\nImage Data Shape: ", image_data[0].shape)

# Normalize pixel values and reshape
image_data = (image_data - np.mean(image_data)) / np.std(image_data)
image_data = image_data.reshape(-1, 128, 128, 1).astype('float32')

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
    RandomRotation(0.1),
    RandomFlip("horizontal"),
    RandomZoom(0.1)
])

# Create tf datasets
batch_size = 32

train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
train = train.map(lambda x, y: (augment_data(x, training=True), y), num_parallel_calls=2)
train = train.prefetch(1)

val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(1)
test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(1)

# Function to handle model creation
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    base_model.trainable = False
    
    # Model Arcitecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=outputs)

# Callback for early stopping
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.01)
]

# Training parameters
epochs = 10
initial_lr = 0.001

# Train model with transfer learning
print("\nTraining MobileNetV2 with Transfer Learning 1st Stage...")
transfer_model = create_model()
transfer_model.compile(optimizer=Adam(learning_rate=initial_lr), loss='binary_crossentropy', metrics=['accuracy'])

transfer_model_history = transfer_model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=1)

# Evaluate model
print("Evaluating Model on Test Set...")
transfer_model_results = transfer_model.evaluate(test, verbose=1)

# Save Transfer learning model
transfer_model.save('results/sota/model/final_transfer_model.h5')

# Print test results
print("\nTransfer Learning Model Test Results:")
print(f"Loss: {transfer_model_results[0]:.4f}")
print(f"Accuracy: {transfer_model_results[1]:.4f}")

# {lot model history
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(transfer_model_history.history['accuracy'], label='Training')
plt.plot(transfer_model_history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(transfer_model_history.history['loss'], label='Training')
plt.plot(transfer_model_history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f'{generated_image_dir}/training_history.png')
plt.close()

# Save training history and results to file
with open(results_file, 'w') as f:
    f.write("MobileNetV2 Transfer Learning Test Results\n")
    f.write("================================\n\n")
    f.write(f"Test Loss: {transfer_model_results[0]:.4f}\n")
    f.write(f"Test Accuracy: {transfer_model_results[1]:.4f}\n")