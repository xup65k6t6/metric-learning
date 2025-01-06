import os
import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from utils.dataload import StructuredDataset
from utils.env_setup import check_device, setup_environment
from utils.img_process import (
    ImageDataset,
    calculate_mean_std,
    get_albumentations_transforms,
    visualize_batch,
)
from utils.model_setup import (
    EarlyStopping,
    EmbeddingModel,
    LiftedStructureLoss,
    compute_distances,
    evaluate_embeddings_strctured,
    split_df,
)


def train_lifted_structure_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs=50,
    model_save_path="/runs/best_lsl_model.pt",
    mlflow_tracking_uri=None,
    experiment_name="ContrastiveLearning",
    patience=10,
    lr_patience=5,
    load_pretrained_model=False,
    run_name=None,
):
    if load_pretrained_model:
        model.load_state_dict(
            torch.load(model_save_path, map_location=device, weights_only=True)
        )
        print(f"Loaded pretrained model from {model_save_path}")
        return model

    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=model_save_path
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=lr_patience, verbose=True
    )

    # lr_scheduler = setup_scheduler(optimizer, epochs, lrf = 0.1)
    # lr_scheduler = setup_scheduler(optimizer, epochs, lrf = 0.1 , warmup_epochs = int(epochs ** .5))

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "initial_learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": (
                    train_loader.batch_size
                    if hasattr(train_loader, "batch_size")
                    else "unknown"
                ),
                "loss_function": loss_fn.__class__.__name__,
                "loss_margin": loss_fn.margin,
                "optimizer": optimizer.__class__.__name__,
                "model_architecture": model.__class__.__name__,
                "early_stopping_patience": patience,
                "lr_scheduler_patience": lr_patience,
                "device": device,
                "model_save_path": model_save_path,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
            }
        )

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                embeddings = model(images)
                loss = loss_fn(embeddings, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            model.eval()
            total_val_loss = 0
            ebd_lst = []
            lbl_lst = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    embeddings = model(images)
                    ebd_lst.append(embeddings)
                    lbl_lst.append(labels)
                    loss = loss_fn(embeddings, labels)
                    total_val_loss += loss.item()

            # lists to tensors
            ebd_lst = torch.cat(ebd_lst, dim=0)
            lbl_lst = torch.cat(lbl_lst, dim=0)

            ebd_distances = compute_distances(ebd_lst, lbl_lst)
            mlflow.log_metrics(ebd_distances, step=epoch)

            avg_val_loss = total_val_loss / len(val_loader)

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            lr_scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model, device)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load the best model based on early stopping
    model = early_stopping.load_best_model(model)
    print(f"Training complete. Best validation loss: {early_stopping.val_loss_min:.4f}")
    # mlflow.pytorch.log_model(model, artifact_path="model")
    mlflow.autolog(disable=True)
    return model


def train(
    df: pd.DataFrame,
    unfreeze_layers: bool = False,
    margin=1.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    model_save_path="/best_lsl_model.pt",
    load_pretrained_model=False,
    top_k=max(10, int(0.8 * 16)),
    epochs=50,
    run_name=None,
):
    print(f"Using device: {device}")
    train_df, val_df = split_df(df, train_size=0.8, seed=42)
    mean, std = calculate_mean_std(train_df["img_path"].tolist())
    # transform = get_transforms()
    transform = get_albumentations_transforms(augment=True, mean=mean, std=std)
    transofrm_val = get_albumentations_transforms(augment=False, mean=mean, std=std)
    train_dataset = ImageDataset(df=train_df, transform=transform, duplicate_factor=5)
    val_dataset = ImageDataset(df=val_df, transform=transofrm_val, duplicate_factor=1)

    # Prepare dataset
    structured_train_dataset = StructuredDataset(train_dataset)
    structured_train_loader = DataLoader(
        structured_train_dataset, batch_size=100, shuffle=True
    )
    structured_val_dataset = StructuredDataset(val_dataset)
    structured_val_loader = DataLoader(
        structured_val_dataset, batch_size=100, shuffle=True
    )

    model = EmbeddingModel(unfreeze_layers=unfreeze_layers).to(device)
    loss_fn = LiftedStructureLoss(margin=margin, top_k=top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    trained_model = train_lifted_structure_model(
        model,
        structured_train_loader,
        structured_val_loader,
        optimizer,
        loss_fn,
        device,
        epochs=epochs,
        experiment_name="ContrastiveLearning",
        patience=10,
        lr_patience=5,
        model_save_path=model_save_path,
        load_pretrained_model=load_pretrained_model,
        run_name=run_name,
    )

    # Evaluate embeddings
    embedding_metrics = evaluate_embeddings_strctured(
        trained_model, structured_val_loader, device
    )
    print("Embedding Evaluation Metrics:", embedding_metrics)
    return (
        trained_model,
        embedding_metrics,
        structured_train_loader,
        structured_val_loader,
    )

def download_mnist(save_folder_dir="/data/mnist", csv_name="mnist.csv"):
    # Download MNIST dataset
    mnist_train = datasets.MNIST(
        root=save_folder_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    mnist_test = datasets.MNIST(
        root=save_folder_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Initialize lists for storing metadata
    train_data = []
    test_data = []

    # Save training images and create metadata
    for idx, (img, label) in enumerate(mnist_train):
        img_folder = os.path.join(save_folder_dir, "train")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, f"img_{idx}.png")
        transforms.ToPILImage()(img).save(img_path)
        train_data.append({"img_path": img_path, "label": int(label)})

    # Save test images and create metadata
    for idx, (img, label) in enumerate(mnist_test):
        img_folder = os.path.join(save_folder_dir, "test")
        os.makedirs(img_folder, exist_ok=True)
        img_path = os.path.join(img_folder, f"img_{idx}.png")
        transforms.ToPILImage()(img).save(img_path)
        test_data.append({"img_path": img_path, "label": int(label)})

    # Combine training and testing data into a DataFrame
    df = pd.DataFrame(train_data + test_data)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(save_folder_dir, csv_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    return df



def main():
    device = check_device()
    folder_dir, tracking_uri = setup_environment(device_type=device.type)
    print(f"Folder directory: {folder_dir}")
    print(f"Tracking URI: {tracking_uri}")

    # data
    # <TODO>: load your dataset into a dataframe here. Below takes MNIST dataset as an example
    # download MNIST dataset

    df = download_mnist(save_folder_dir= os.path.join(folder_dir, "data", "mnist"), csv_name="mnist.csv")

    # df = pd.read_csv(folder_dir + "/data/yours.csv")
    # df["img_path"] = folder_dir + "/data/" + df["img_path"]

    # mean, std = calculate_mean_std(df["img_path"].tolist())
    # transform = get_albumentations_transforms(
    #     img_size=(224, 224), augment=True, mean=mean, std=std
    # )
    # test_dataset = ImageDataset(df=df, transform=transform, duplicate_factor=1)
    # visualize_batch(test_dataset, num_images=10, show=True)

    # model
    (
        trained_model_structured,
        embedding_metrics_structured,
        structured_train_loader,
        structured_val_loader,
    ) = train(
        df=df,
        unfreeze_layers=True,
        margin=1,
        device=device,
        model_save_path=os.path.dirname(folder_dir) + "/runs/best_lsl_model.pt",
        top_k=max(10, int(0.8 * 16)),  # select top k loss
        epochs=50,
        run_name="LiftedStructureLoss",
        # load_pretrained_model = True
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
