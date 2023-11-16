# Imports librerías
# TF 2.13
import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import numpy as np
import math

import tensorflow.keras.applications.efficientnet_v2 as effnV2
from tensorflow.keras.layers import Dense, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout
from sklearn.utils.class_weight import compute_class_weight

# Definición métodos de creación del dataset
dict_map_class = {
    'Loose Silky-bent': 0,
    'Common Chickweed': 1,
    'Scentless Mayweed': 2,
    'Small-flowered Cranesbill': 3,
    'Fat Hen': 4,
    'Charlock': 5,
    'Sugar beet': 6,
    'Cleavers': 7,
    'Black-grass': 8,
    'Shepherds Purse': 9,
    'Common wheat': 10,
    'Maize': 11,   
}


def get_dict_dataset(
    dataset_path: str = 'dataset'
):
    """
    Creates a dictionary mapping image paths to tuples containing class and partition.

    Args:
        dataset_path (str): Path to the dataset directory. Defaults to 'dataset'.

    Returns:
        dict: A dictionary where keys are image paths and values are tuples (class, partition).
    """

    dict_dataset = {}

    train_classes = os.listdir(os.path.join(dataset_path, 'train'))

    for train_class in train_classes:
        class_path = os.path.join(dataset_path, 'train', train_class)
        train_imgs = os.listdir(class_path)
        
        # Split Train images to a 80% for a Train Split for each class
        for train_img in train_imgs[:int(len(train_imgs)*0.8)]:
            train_img_path = os.path.join(class_path, train_img)
            dict_dataset[train_img_path] = (dict_map_class[train_class], 'Train')

        # Assign the rest 20% to Valid Split for each class
        for valid_img in train_imgs[int(len(train_imgs)*0.8):]:
            valid_img_path = os.path.join(class_path, valid_img)
            dict_dataset[valid_img_path] = (dict_map_class[train_class], 'Valid')

    # Geting Test Images
    test_path = os.path.join(dataset_path, 'test')
    test_imgs = os.listdir(test_path)

    for test_img in test_imgs:
        test_img_path = os.path.join(test_path, test_img)
        dict_dataset[test_img_path] = ("Unkown", 'Test')

    return dict_dataset


def dict2dataframe(
    input_dict: Dict[str, Tuple]
):
    """
    Converts a dictionary to a pandas DataFrame with columns for 'path', 'label', and 'split'.

    Args:
        input_dict (dict): A dictionary where keys are image paths and values are tuples (label, split).

    Returns:
        pd.DataFrame: A DataFrame with columns 'path', 'label', and 'split'.
    """
    df = pd.DataFrame([(key, values[0], values[1]) for key, values in input_dict.items()], columns=['path', 'label', 'split'])

    # Returns shuffled datasets
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# Creación TF Datasets

def tf_augmenter():
    """
    Returns a TensorFlow function for data augmentation.

    The returned function applies random transformations to the input images,
    including random flips (up-down and left-right), random brightness adjustments,
    and random contrast adjustments.

    Returns:
        callable: A TensorFlow function that takes a variable number of arguments (tensors)
                  representing the dataset and applies data augmentation to the images.
    """
    @tf.function
    def f(*dataset):
        output= list(dataset)
        image = output[0]
        
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_flip_up_down(image)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.5:
            image = tf.image.random_brightness(image, 0.15)
        if tf.random.uniform([1], minval=0, maxval=1) > 0.7:
            image = tf.image.random_contrast(image, 0.6, 1.4)

        output[0] = image
        return output
    return f


@tf.function
def load_image(*inputs):
    """
    TensorFlow function to load an image using a numpy function.

    Args:
        *inputs: Variable number of input tensors.

    Returns:
        list: A list of output tensors with the loaded image as the first element.
    """
    outputs = list(inputs)
    image = tf.numpy_function(load_image_np, [inputs[0]], tf.float32)
    image.set_shape([None, None, 3])
    outputs[0] = image
    
    return outputs


def load_image_np(path):
    """
    Loads an image from the specified path and convert it to a NumPy array.

    Args:
        path (str): The path to the image file.

    Returns:
        np.ndarray: A NumPy array representing the loaded image in RGB format.
    """
    return np.array(Image.open(path).convert('RGB')).astype(np.float32)


def resize(index=0, resize_to=None):
    """
    Returns a TensorFlow function to resize an image in a dataset.

    Args:
        index (int): Index of the image tensor in the dataset. Defaults to 0.
        resize_to (tuple or list or None): Target size for resizing. If None, no resizing is performed. Defaults to None.

    Returns:
        callable: A TensorFlow function that resizes the image in the dataset.
    """
    def f(*dataset):
        output = list(dataset)
        resized_image = tf.image.resize(dataset[index], resize_to)
        resized_image = tf.cast(resized_image, tf.uint8)
        output[index] = resized_image
        
        return output
    return f


def preprocess_input(index):
    """
    Returns a TensorFlow function to preprocess an image in a dataset.

    Args:
        index (int): Index of the image tensor in the dataset.

    Returns:
        callable: A TensorFlow function that preprocesses the image in the dataset.
    """
    @tf.function
    def f(*dataset):
        output = list(dataset)
        image = dataset[index]
        image = tf.cast(image, tf.float32)
        image = image / 255.
        output[index] = image
        
        return output
    return f



def get_dataset(
    df: pd.DataFrame,
    input_size: Tuple[int, int],
    shuffle: bool = False,
    batch_size: int = None,
    gray_scale: bool = False,
    augmenter: bool = False,
    num_aug: int = None,
    test_set: bool = False
)->tf.data.Dataset:
    """
    Creates a TensorFlow dataset from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing information about the dataset.
        input_size (Tuple[int, int]): Tuple representing the target size for image resizing.
        shuffle (bool): Whether to shuffle the dataset. Defaults to False.
        batch_size (int): Batch size for the dataset. If None, no batching is performed. Defaults to None.
        gray_scale (bool): Whether to convert images to grayscale. Defaults to False.
        augmenter (bool): Whether to apply data augmentation. Defaults to False.
        num_aug (int): Number of augmentations to apply if augmenter is True. Defaults to None.
        test_set (bool): Whether the dataset is a test set. Defaults to False.

    Returns:
        tf.data.Dataset: A TensorFlow dataset prepared based on the provided options.
    """
    # Prints info about labels distribution 
    print('Number of instances per label: ',
          pd.Series(df['label']).value_counts(), sep='\n')
    print('\nPercentaje of instances per label: ',
          pd.Series(df['label']).value_counts().div(pd.Series(df['label']).shape[0]),
          sep='\n')

    names = np.array(df['path'], dtype=str)

    if not test_set:
        labels = np.array(tf.keras.utils.to_categorical(df['label'], num_classes=12))
    else:
        labels = np.ones(len(names))

    data = names, labels

    # Creates a tf dataset from paths and labels
    dataset = tf.data.Dataset.from_tensor_slices(data)

    # Shuffles the entire dataset
    if shuffle:
        print(' > Shuffle')
        dataset = dataset.shuffle(len(names))

    # Loading Images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Resize to desired size
    dataset = dataset.map(resize(resize_to=input_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Augmentation
    if augmenter:
        print(f' > Augmentamos datos numero {num_aug}')
        if num_aug == 1:
            dataset = dataset.map(tf_augmenter(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Preprocessing input
    dataset = dataset.map(preprocess_input(0), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Converts to gray Scale
    if gray_scale:
        print(' > Escala de grises')
        dataset = dataset.map(lambda *args: (tf.image.rgb_to_grayscale(args[0]), *args[1:]))

    # Prepare batch_size
    if batch_size is not None:
        print(' > Establecemos el batchsize')
        dataset = dataset.batch(batch_size)
    
    # Prefetch to overlap data preprocessing and model execution
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset