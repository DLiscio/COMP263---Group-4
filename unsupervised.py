"""
COMP263 - Group 4: Evaluating Deep Neural Networks using the Histopathologic Cancer Detection dataset
Unsupervised Learning
"""
import pandas as pd
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Conv2D, Dropout, Flatten, SpatialDropout2D
from tensorflow.keras.optimizers import Adam

# Ensure directories/paths exists
image_dir = 'data/train/'
labels_file = 'data/train_labels.csv'
reduced_image_dir = 'data/reduced_train/'
os.makedirs(reduced_image_dir, exist_ok=True)

current_time = time.strftime("%Y%m%d-%H%M%S")
results_file = f'results/unsupervised/run_logs/GAN_Model_summary_{current_time}.txt'
generated_image_dir = f'results/unsupervised/images/run_{current_time}'
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

# Batch and shuffle training data and declare model parameters
batch_size = 64
train = tf.data.Dataset.from_tensor_slices(image_data).shuffle(20000).batch(batch_size, drop_remainder=True)
latent_dim = 128
epochs = 50

# Build the generator of GAN
generator_model = Sequential([
    Input(shape=(latent_dim,)),
    Dense(24*24*256, use_bias=False),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    Reshape((24,24,256)),
    Conv2DTranspose(128, kernel_size=(5,5), strides=(2,2), padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(64, kernel_size=(5,5), strides=(2,2), padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(32, kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False),
    BatchNormalization(),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(1, kernel_size=(3,3), strides=(1,1), padding="same", activation="tanh", use_bias=True)
])

# Build the discriminator
discriminator_model = Sequential([
    Input(shape=(96,96,1)),
    Conv2D(64, kernel_size=(5,5), strides=(2,2), padding="same"),
    LeakyReLU(alpha=0.2),
    SpatialDropout2D(0.3),
    Conv2D(128, kernel_size=(5,5), strides=(2,2), padding="same"),
    LeakyReLU(alpha=0.2),
    SpatialDropout2D(0.3),
    Conv2D(128, kernel_size=(3,3), strides=(2,2), padding="same"),
    LeakyReLU(alpha=0.2),
    Flatten(),
    Dense(1, activation="sigmoid")
])

# Create loss function
loss = keras.losses.BinaryCrossentropy(from_logits=False)

# Create the optimizers
initial_lr = 0.0002
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=300,  
    decay_rate=0.95
)
generator_optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=lr_schedule, beta_1=0.5)

# Save Generator and Discriminator models summary to file
with open(results_file, 'w', encoding='utf-8' ) as f:
    # Header
    f.write("="*50 + "\n")
    f.write("GAN FINAL MODEL TRAINING CONFIGURATION\n")
    f.write("="*50 + "\n\n")

    # Training Parameters
    f.write("-"*20 + " Training Parameters " + "-"*20 + "\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Latent Dimension: {latent_dim}\n")
    f.write(f"Epochs: {epochs}\n\n")

    # Loss and Optimizers
    f.write("-"*20 + " Loss & Optimizers " + "-"*20 + "\n")
    f.write(f"Loss Function: {loss.__class__.__name__}\n")
    
    # Format optimizer config for better readability
    gen_opt_config = generator_optimizer.get_config()
    disc_opt_config = discriminator_optimizer.get_config()
    
    f.write("\nGenerator Optimizer (Adam):\n")
    f.write(f"Learning Rate: {gen_opt_config['learning_rate']}\n")
    f.write(f"Beta 1: {gen_opt_config['beta_1']}\n")
    
    f.write("\nDiscriminator Optimizer (Adam):\n")
    f.write(f"Learning Rate: {disc_opt_config['learning_rate']}\n")
    f.write(f"Beta 1: {disc_opt_config['beta_1']}\n")

    # Model Architectures
    f.write("="*50 + "\n")
    f.write("MODEL ARCHITECTURES\n")
    f.write("="*50 + "\n\n")
    
    f.write("-"*20 + " Generator Architecture " + "-"*20 + "\n")
    generator_model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n")
    
    f.write("-"*20 + " Discriminator Architecture " + "-"*20 + "\n")
    discriminator_model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n")

    # Footer
    f.write("="*50 + "\n")
    f.write("TRAINING RESULTS FOLLOW BELOW\n")
    f.write("="*50 + "\n")

# Define training function
@tf.function
def training_step(images):
    noise_factor = 0.005
    image_noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_factor)
    noisy_images = images + image_noise

    # Random flipping for data augmentation
    noisy_images = tf.image.random_flip_left_right(noisy_images)
    noisy_images = tf.image.random_flip_up_down(noisy_images)

    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)

        # Add noise to generated images 
        gen_noise = tf.random.normal(shape=tf.shape(generated_images), mean=0.0, stddev=noise_factor)
        noisy_generated = generated_images + gen_noise
        
        real_output = discriminator_model(noisy_images, training=True)
        fake_output = discriminator_model(noisy_generated, training=True)
        
        # Label Smoothing
        real_labels = tf.random.uniform([batch_size, 1], 0.8, 0.9)
        fake_labels = tf.random.uniform([batch_size, 1], 0.0, 0.1)

        # Loss
        gen_loss = loss(real_labels, fake_output)  
        disc_loss = loss(real_labels, real_output) + loss(fake_labels, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    # Clip gradients
    gradients_of_generator = [tf.clip_by_norm(g, 1.0) for g in gradients_of_generator]
    gradients_of_discriminator = [tf.clip_by_norm(g, 1.0) for g in gradients_of_discriminator]

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    return gen_loss, disc_loss

# Model Training
def train_models(dataset):
    print("########## GAN Model Training ##########")

    # Lists to store losses 
    gen_losses_history = []
    disc_losses_history = []
    epochs_list = []

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write("\nEpoch, Time(s), Generator_Loss, Discriminator_Loss\n") 
    
        for epoch in range(epochs):
            epoch_start = time.time()

            gen_losses = []
            disc_losses = []

            for img_batch in dataset:
                g_loss, d_loss = training_step(img_batch)
                gen_losses.append(float(g_loss))
                disc_losses.append(float(d_loss))
            
            epoch_time = time.time() - epoch_start
            avg_gen_loss = np.mean(gen_losses)
            avg_disc_loss = np.mean(disc_losses)

            gen_losses_history.append(avg_gen_loss)
            disc_losses_history.append(avg_disc_loss)
            epochs_list.append(epoch + 1)

            print(f"Epoch {epoch+1}/{epochs}: Completed in {epoch_time:.2f}s, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            f.write(f"{epoch+1}, {epoch_time:.2f}, {avg_gen_loss:.4f}, {avg_disc_loss:.4f}\n")

            # Generate and save sample image every 10 epochs, including first epoch
            if (epoch + 1) % 10 == 0 or epoch == 0:
                noise = tf.random.normal([1, latent_dim])
                generated_img = generator_model(noise, training=False)
                generated_img = 0.5 * generated_img + 0.5
                
                plt.figure(figsize=(8,8))
                plt.imshow(generated_img[0,:,:,0], cmap='gray')
                plt.axis("off")
                plt.savefig(f"{generated_image_dir}/epoch_{epoch+1}.png")
                plt.close()

    # Plot losses
    plt.figure(figsize=(8, 8))
    plt.plot(epochs_list, gen_losses_history, 'r-', label='Generator Loss')
    plt.plot(epochs_list, disc_losses_history, 'b-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss Over Time')
    plt.legend()
    plt.savefig(f"{generated_image_dir}/loss_vs_epochs_final.png")
    plt.close()

# Train the models
train_models(train)

# Create 12 sample vectors 
trained_sample_vectors = tf.random.normal([12, latent_dim])

# Generate images from model
generated_images = generator_model(trained_sample_vectors, training=False)

# Normalize pixel values
generated_images = ((generated_images / 127.5) + 127.5)

# Plot and save the generated images
fig = plt.figure(figsize=(8,8))
for i in range(12):
    # Save image for future testing with supervised model
    img = generated_images[i,:,:,0].numpy() 
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(f"results/unsupervised/generated_test_data/generated_sample_{i+1}.tif", img)

    plt.subplot(4,3, i+1)
    plt.axis("off")
    plt.imshow(generated_images[i,:,:,0], cmap='gray')
plt.savefig(f"{generated_image_dir}/generated_images_Plot_Model_Final.png")
plt.close(fig)