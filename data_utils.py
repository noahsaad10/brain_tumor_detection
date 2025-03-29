import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np


def show_images(folder, title, num_images=4):
    images = random.sample(os.listdir(folder), num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(title)
    
    plt.show()


def preprocess_and_label_data(directory: str, label: int, IMG_SIZE: int) -> tuple:
    """
    Preprocess and label the directory

    Args:
        directory (str): directory path
        label (int): label to add to images

    Returns:
        tuple: the preprocessed images and labels
    """
    
    images = []
    labels = []

    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(label)
    return images, labels

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply((img * 255).astype(np.uint8)) / 255.0

def plot_model_info(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.show()
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.show()

def predict_image(model, img_path, img_size):
    # Load and preprocess the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = apply_clahe(np.array(img))
    img = img.reshape(-1, img_size, img_size, 1)
    
    # Predict
    pred = model.predict(img)
    
    # Display the image
    plt.imshow(img.reshape(img_size, img_size), cmap='gray')
    plt.axis('off')
    
    # Display the prediction
    if pred < 0.5:
        plt.title(f"Predicted: No Tumor ({(1 - pred.squeeze())*100:.2f}% confidence)")
    else:
        plt.title(f"Predicted: Tumor ({pred.squeeze()*100:.2f}% confidence)")

    plt.show()


