# Import required libraries and modules
import os
from torch.utils.data import Dataset
from PIL import Image
import json

# Define the custom dataset class for the ImageNet dataset
class MyImageNetDataset(Dataset):
    def __init__(self, root, split, transform=None):
        #root = "C:/ImagenetData/ILSVRC/Data/CLS-LOC"
        self.samples = []  # List to store paths to images
        self.targets = []  # List to store corresponding class labels for images
        self.transform = transform  # Transformation function to apply to images (e.g., normalization)
        self.syn_to_class = {}  # Dictionary to map synset IDs to class IDs

        # Load class index file and create a mapping from synset IDs to class IDs
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        # Load validation labels file and create a mapping from image file names to synset IDs
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        # Construct the path to the directory containing the images for the specified split (train/val)
        samples_dir = os.path.join(root, "", split)
        
        # Iterate over the directory to populate the samples and targets lists with image paths and class labels
        for entry in os.listdir(samples_dir):
            if split == "train":
                # For training images, use the directory name as the synset ID
                syn_id = entry
                target = self.syn_to_class[syn_id]  # Map synset ID to class ID
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                # For validation images, use the mapping from image file name to synset ID
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]  # Map synset ID to class ID
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Retrieve an image and its corresponding label by index, applying any transformations if specified
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            x = self.transform(x)  # Apply transformations to the image
        return x, self.targets[idx]  # Return the transformed image and its label
