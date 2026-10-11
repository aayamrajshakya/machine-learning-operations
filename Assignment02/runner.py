# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import optuna
import mlflow
import mlflow.pytorch
from PIL import Image
from tqdm import tqdm

DATASET_PATH = "sea_animal_dataset"  # https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste/data
EPOCHS = 6
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
OPTUNA_TRIALS = 11
MODEL_NAME = "mobilenet_model"


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_dataloaders():
    # create img dataset from DATASET_PATH
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    n = len(full_dataset)

    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(123),
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(123),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, full_dataset.classes


def create_model(trial, num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers and update only the very last ones
    for p in model.parameters():
        p.requires_grad = False

    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)

    context = torch.enable_grad() if is_train else torch.inference_mode()
    running_loss, correct, total = 0.0, 0, 0

    with context:
        progress_bar = tqdm(loader, desc=desc, leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

            progress_bar.set_postfix(
                loss=f"{running_loss / total:.4f}", acc=f"{correct / total:.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def objective(trial, train_loader, val_loader, num_classes):
    model = create_model(trial, num_classes).to(DEVICE)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "Adam":
        optimizer = optim.Adam(trainable_params, lr=lr)
    else:
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(nested=True) as run:
        # Store run_id in the Optuna trial to retrieve the best model later
        trial.set_user_attr("run_id", run.info.run_id)
        mlflow.log_params(trial.params)

        for epoch in range(EPOCHS):
            print(f"\nTrial {trial.number} | Epoch {epoch+1}/{EPOCHS}")
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, desc="Training"
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, desc="Validation"
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Log the trained model to MLflow
        input_example = torch.randn(1, 3, 224, 224).numpy()
        mlflow.pytorch.log_model(
            model, "model", input_example=input_example, serialization_format="pickle"
        )

        return val_acc


def inference(image_path, model, class_names):
    model = model.to(DEVICE).eval()
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        pred = model(x).argmax(dim=1).item()
    return class_names[pred]


def main():
    torch.manual_seed(123)
    if not os.path.isdir(DATASET_PATH):
        print(f"Dataset not found at '{DATASET_PATH}'.")
        return

    mlflow.set_experiment(experiment_name="transfer-learning-w-mobilenet")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=123)
    )

    with mlflow.start_run(run_name="Hyperparameter tuning w/ Optuna"):
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, len(class_names)),
            n_trials=OPTUNA_TRIALS,
        )
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_acc", study.best_value)

    print(f"\nBest val_acc: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_trial = study.best_trial

    best_run_id = best_trial.user_attrs.get("run_id")
    if best_run_id:
        best_model_uri = f"runs:/{best_run_id}/model"

        # Register the best model to the MLflow model registry
        print("\nRegistering the best model to the model registry...")
        client = mlflow.MlflowClient()
        model_version = mlflow.register_model(model_uri=best_model_uri, name=MODEL_NAME)

        # Assign the 'production' alias to this version
        client.set_registered_model_alias(
            MODEL_NAME, "production", model_version.version
        )
        print("Model successfully registered for production")

        # Load the registered model and evaluate it on the held-out test set
        production_model = mlflow.pytorch.load_model(
            f"models:/{MODEL_NAME}@production"
        ).to(DEVICE)
        _, test_acc = run_epoch(
            production_model, test_loader, nn.CrossEntropyLoss(), desc="Test"
        )
        print(f"Test acc: {test_acc:.4f}")
        client.log_metric(best_run_id, "test_acc", test_acc)

        # Test inference on a sample image
        sample_class = class_names[0]
        sample_dir = os.path.join(DATASET_PATH, sample_class)
        image_files = [
            f
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if image_files:
            sample_image = os.path.join(sample_dir, image_files[0])
            prediction = inference(sample_image, production_model, class_names)
            print(f"Predicted: {prediction} | Actual: {sample_class}")


if __name__ == "__main__":
    main()

# %%
