import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def unnormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses the normalization of an image tensor."""
    img_tensor = img_tensor.clone()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse normalization
    return img_tensor


def plot_tsne(embeddings, labels, title="t-SNE Visualization"):
    """
    Create t-SNE plot of embeddings with improved class labeling

    Args:
    - embeddings: numpy array of feature vectors
    - labels: corresponding class labels
    """
    # Ensure embeddings and labels are NumPy arrays
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    unique_labels = np.unique(labels)
    color_map = plt.colormaps["tab20"]

    # Plot each class with a different color
    for i, label in enumerate(unique_labels):
        # Find indices for this specific label
        mask = labels == label
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=[color_map(i / len(unique_labels))],
            label=f"Class {label}",
            alpha=0.7,
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Feature 1", fontsize=12)
    plt.ylabel("t-SNE Feature 2", fontsize=12)

    plt.legend(title="Classes", title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.show()

    class_counts = {}
    for label in unique_labels:
        class_counts[label] = np.sum(labels == label)
    print("Class Distribution:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")


def visualize_embeddings(
    model, extract_ebd_fn, dataloader, device, title="Embedding Visualization"
):
    """
    Extract and visualize embeddings

    Args:
    - model: Trained neural network model
    - extract_ebd_fn: Function to extract embeddings
    - dataloader: PyTorch DataLoader
    - device: Computing device
    """
    embeddings, labels = extract_ebd_fn(model, dataloader, device)
    plot_tsne(embeddings, labels, title=title)


def visualize_with_logistic_regression(
    dataset,
    model,
    classifier,
    class_dict,
    num_samples=5,
    device="cpu",
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    show=True,
):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        image, label = dataset[i]

        if torch.is_tensor(image):
            org_image = unnormalize(image.clone(), mean=mean, std=std)
            org_image = org_image.permute(1, 2, 0).numpy()
            org_image = (org_image * 255).astype(
                "uint8"
            )  # Convert to 8-bit image for display

        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            embeddings = model(image_tensor)
            feature_vector = embeddings.cpu().numpy().flatten()
            prediction = classifier.predict([feature_vector])[0]
            color = "green" if prediction == label else "red"

        axes[i].imshow(org_image)
        axes[i].set_title(
            f"Label: {class_dict[label]} \nPrediction: {class_dict[prediction]}",
            bbox=dict(facecolor=color, alpha=0.5, edgecolor="none", pad=2),
        )
        axes[i].axis("off")

    plt.tight_layout()
    if show:
        plt.show()


# example:
# visualize_with_logistic_regression(structured_val_loader.dataset, model= trained_model_structured, classifier= classifier,class_dict =class_dict, num_samples=10, device=device, mean = mean, std = std)
