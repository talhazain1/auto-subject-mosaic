import os
import pandas as pd
import tensorflow as tf

# Set the DUTS dataset root directory
dataset_root = '/Users/TalhaZain/Image_Segmentation/archive'
metadata_path = os.path.join(dataset_root, 'metadata.csv')
metadata = pd.read_csv(metadata_path)

# Split metadata into training and testing sets
train_df = metadata[metadata['split'] == 'train']
test_df  = metadata[metadata['split'] == 'test']

# Build full paths using the 'image_path' and 'mask_path' columns
train_image_paths = [os.path.join(dataset_root, path) for path in train_df['image_path'].tolist()]
train_mask_paths  = [os.path.join(dataset_root, path) for path in train_df['mask_path'].tolist()]
test_image_paths  = [os.path.join(dataset_root, path) for path in test_df['image_path'].tolist()]
test_mask_paths   = [os.path.join(dataset_root, path) for path in test_df['mask_path'].tolist()]

print("Number of training images:", len(train_image_paths))
print("Number of testing images:", len(test_image_paths))

# Function to load and preprocess images and masks on the fly
def load_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    # Read and decode the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0

    # Read and decode the mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, target_size)
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask

# Create a tf.data.Dataset
def create_tf_dataset(image_paths, mask_paths, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))
    ds = ds.map(lambda img, msk: load_image_and_mask(img, msk),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = create_tf_dataset(train_image_paths, train_mask_paths, batch_size=8, shuffle=True)
test_dataset = create_tf_dataset(test_image_paths, test_mask_paths, batch_size=8, shuffle=False)

# ----- Simple U-Net Model Definition -----
from tensorflow.keras import layers, models

def simple_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D((2, 2))(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = simple_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ----- Training the Model -----
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# ----- Saving the Trained Model -----
model.save('unet_model.h5')
print("Model saved as unet_model.h5")
