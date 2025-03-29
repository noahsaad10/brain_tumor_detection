import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(data_set_dir: str) -> None:
    """
    Split the dataset into train and test sets

    Args:
        data_set_dir (str): The directory where the dataset is stored
    """

    # Define the dataset directory and output directories
    dataset_dir = data_set_dir
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'

    # Get all file names from the dataset directory
    file_names = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

    # Split the dataset into 90% train and 10% test
    train_files, test_files = train_test_split(file_names, test_size=0.1, random_state=42)

    # Move files to the train directory
    for file_name in train_files:
        shutil.copy(os.path.join(dataset_dir, file_name), os.path.join(train_dir, file_name))

    # Move files to the test directory
    for file_name in test_files:
        shutil.copy(os.path.join(dataset_dir, file_name), os.path.join(test_dir, file_name))

    print("Dataset split completed!")