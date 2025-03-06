import os
import pandas as pd

# Set the root directory of the DUTS dataset.
# This folder should contain 'metadata.csv' and the subfolders.
dataset_root = '/Users/TalhaZain/Image_Segmentation/archive'

# Load the metadata.csv file
metadata_path = os.path.join(dataset_root, 'metadata.csv')
metadata = pd.read_csv(metadata_path)

# For debugging: print header info and unique split values
print("Columns in metadata:", metadata.columns.tolist())
print("Unique split values:", metadata['split'].unique())

# Filter the metadata for training and testing splits
train_df = metadata[metadata['split'] == 'train']
test_df  = metadata[metadata['split'] == 'test']

# Use the image_path and mask_path columns to build full file paths
train_image_paths = [os.path.join(dataset_root, path) for path in train_df['image_path'].tolist()]
train_mask_paths  = [os.path.join(dataset_root, path) for path in train_df['mask_path'].tolist()]
test_image_paths  = [os.path.join(dataset_root, path) for path in test_df['image_path'].tolist()]
test_mask_paths   = [os.path.join(dataset_root, path) for path in test_df['mask_path'].tolist()]

# Output the counts for verification
print("Number of training images:", len(train_image_paths))
print("Number of testing images:", len(test_image_paths))
