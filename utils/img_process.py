import cv2
import random
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def calculate_mean_std(img_paths, img_size=(224, 224)):
    """
    Calculate mean and std for image dataset
    Args:
        img_paths (list): List of image file paths
        img_size (tuple): Target image size for resizing
    Returns:
        mean (list): Mean for each RGB channel
        std (list): Std for each RGB channel
    """
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0

    def process_image(img_path):
        nonlocal pixel_sum, pixel_squared_sum, num_pixels
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0

        num_pixels_local = img.shape[0] * img.shape[1]
        pixel_sum_local = img.sum(axis=(0, 1))
        pixel_squared_sum_local = (img**2).sum(axis=(0, 1))

        return pixel_sum_local, pixel_squared_sum_local, num_pixels_local

    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(int(cpu_count * 1.5), len(img_paths))

    print(f"Processing {len(img_paths)} images using {optimal_workers} workers...")
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, img_paths),
                total=len(img_paths),
                desc="Processing images",
                unit="img",
            )
        )

    for result in results:
        pixel_sum += result[0]
        pixel_squared_sum += result[1]
        num_pixels += result[2]

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_squared_sum / num_pixels - mean**2)

    return mean.tolist(), std.tolist()


def get_albumentations_transforms(
    img_size=(224, 224), augment=True, mean=None, std=None
):
    """
    Returns Albumentations transformations
    Args:
        img_size (tuple): Target image size (width, height)
        augment (bool): Whether to include data augmentation
        mean (list): Mean for normalization
        std (list): Std for normalization
    Returns:
        A.Compose: Albumentations composed transformations
    """
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transforms_list = []

    if augment:
        transforms_list.extend(
            [
                # A.RandomGamma(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.Transpose(p=0.5),
                # A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                # A.HueSaturationValue(p=0.5),
                # A.RandomBrightnessContrast(contrast_limit=0.1,p=0.5),
                # A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=0.2),
                # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.Resize(
                    height=max(img_size), width=max(img_size), p=1.0
                ),  # Resize to ensure it's large enough
                A.RandomCrop(width=img_size[0], height=img_size[1], p=0.2),
                A.ElasticTransform(p=0.2),
                A.GridDistortion(p=0.2),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ]
        )

    transforms_list.extend(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        transform=None,
        duplicate_factor=1,
        aug_img_size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        """
        Args:
            df (DataFrame): DataFrame containing 'img_path' and 'label' columns.
            transform: Transformation to be applied to images (can be Albumentations or PyTorch transform).
            duplicate_factor (int): How many times each image in the dataset should be duplicated.
        """
        self.df = pd.concat([df] * duplicate_factor, ignore_index=True)
        self.transform = transform
        self.duplicate_factor = duplicate_factor
        self.aug_img_size = aug_img_size
        self.basic_transform = A.Compose(
            [
                A.Resize(aug_img_size[0], aug_img_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgpath = self.df.img_path.iloc[idx]
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.df.label.iloc[idx]

        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=img)
                aug_img = augmented["image"]
            else:  # Assume it is a PyTorch transform
                aug_img = self.transform(img)
        else:
            augmented = self.basic_transform(image=img)
            aug_img = augmented["image"]

        return img, aug_img, label


# pytorch transform
def get_transforms(
    img_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def visualize_batch(dataset, num_images=4, show=True):
    """
    Visualize original and augmented images side-by-side from the SingleMaterialDataset.

    Args:
        dataset (SingleMaterialDataset): The dataset to visualize from.
        num_images (int): The number of images to visualize (each image will have its original and augmented version displayed side-by-side).
    """
    fig, ax = plt.subplots(
        num_images, 2, figsize=(12, num_images * 3)
    )  # Create a grid of (num_images, 2) for side-by-side comparison

    for i in range(num_images):
        idx = random.randint(
            0, len(dataset) - 1
        )  # Randomly select an image from the dataset
        img, aug_img, label = dataset[idx]

        for j, image in enumerate([img, aug_img]):
            # If image is a PyTorch tensor, convert it to NumPy
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

            # If the image is normalized, denormalize it
            if image.max() <= 1.0:  # Assume image is in range [0, 1]
                image = (image * 255).astype("uint8")  # Convert to [0, 255]

            ax[i, j].imshow(image)
            if j == 0:
                ax[i, j].set_title(f"Original (Label: {label})")
            else:
                ax[i, j].set_title(f"Augmented (Label: {label})")

            ax[i, j].axis("off")

    plt.tight_layout()
    if show:
        plt.show()


# Example usage
# visualize_batch(test_dataset, num_images=10)
