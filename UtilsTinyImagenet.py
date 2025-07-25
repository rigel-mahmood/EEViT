from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import models, datasets
import pandas as pd
import os
from torchvision import transforms as T
import sys
import torchvision.transforms as transforms
from randomaug import RandAugment

def prepare_data(mainDir):
    VALID_DIR = os.path.join(mainDir, 'val')
    # Unlike training folder where images are already arranged in sub folders based 
    # on their labels, images in validation folder are all inside a single folder. 
    # Validation folder comes with images folder and val_annotations txt file. 
    # The val_annotation txt file comprises 6 tab separated columns of filename, 
    # class label, x and y coordinates, height, and width of bounding boxes
    val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt', 
                           sep='\t', 
                           header=None, 
                           names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    val_data.head()
        # Create separate validation subfolders for the validation images based on
    # their labels indicated in the val_annotations txt file
    val_img_dir = os.path.join(VALID_DIR, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Display first 10 entries of resulting val_img_dict dictionary
    {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

    # Setup function to create dataloaders for image datasets
    # Create subfolders (if not present) for validation images based on label ,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    fp = open(os.path.join(mainDir, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()

    # Display first 20 entries of resulting dictionary
    dclasses = {k: class_to_name_dict[k] for k in list(class_to_name_dict)[:20]}  
    print(dclasses)
    
    
    
def generate_dataloader(data, name, transform, batch_size=64):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)
    # Set options for device
    use_cuda=True    
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    return dataloader

def get_loaders_tiny_imagenet(mainDir, batch_size, aug=True,N = 2, M = 14):
    # Define transformation sequence for image pre-processing
    # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
    # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
    # std=[0.229, 0.224, 0.225])
    preprocess_transform = T.Compose([
                    # T.Resize(256), # Resize images to 256 x 256
                    # T.CenterCrop(224), # Center crop image
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),  # Converting cropped images to tensors
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 
    ])

    preprocess_transform_pretrain = T.Compose([
                    #T.Resize(256), # Resize images to 256 x 256
                    #T.CenterCrop(224), # Center crop image
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),  # Converting cropped images to tensors
                    #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_train = transforms.Compose([
        transforms.Resize(72),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    N, M = 2, 14  # Example values for RandAugment parameters
  
    transform_test = transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M, 64))
    
    VALID_DIR = os.path.join(mainDir, 'val')
    TRAIN_DIR = os.path.join(mainDir, 'train')
    val_img_dir = mainDir + '/val/images'

    val_loader = generate_dataloader(val_img_dir, "val",
                                 transform=transform_test)
    train_loader = generate_dataloader(TRAIN_DIR, "train",
                                   transform=transform_train)
    return train_loader, val_loader

def main():
    mainDir = "D:/data/TinyImageNet/tiny-imagenet-200" 
    batch_size = 64
    train_loader, val_loader = get_loaders_tiny_imagenet(mainDir, batch_size)
    for i, (x,y) in enumerate(val_loader):
        print(x.shape)
        print(y)
        break

if __name__ == "__main__":
    sys.exit(int(main() or 0))

