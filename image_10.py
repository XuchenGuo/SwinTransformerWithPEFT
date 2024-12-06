# This project utilizes PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.
# For reference, see the official PEFT documentation:
# https://huggingface.co/docs/peft/en/index
#
# Citation:
# @Misc{peft,
#   title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
#   author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
#   howpublished = {\url{https://github.com/huggingface/peft}},
#   year =         {2022}
# }
"""
Constructs a swin_tiny architecture from
`Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/abs/2103.14030>`_.

Args:
    weights (:class:`~torchvision.models.Swin_T_Weights`, optional): The
        pretrained weights to use. See
        :class:`~torchvision.models.Swin_T_Weights` below for
        more details, and possible values. By default, no pre-trained
        weights are used.
    progress (bool, optional): If True, displays a progress bar of the
        download to stderr. Default is True.
    **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
        base class. Please refer to the `source code
        <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
        for more details about this class.

autoclass:: torchvision.models.Swin_T_Weights
    :members:
"""

# Import libraries
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import swin_t
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model  # Requires `peft` library
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import argparse
from torchvision.models import Swin_T_Weights  # Import weights enumeration

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------- Dataset Class ----------------------------------
class ImageByteDataset(Dataset):
    """
    Custom Dataset to load images stored as bytes and associated labels.

    Args:
        image_data (pd.DataFrame): DataFrame with image bytes and labels.
        transform (callable, optional): Transformations to apply to the images.
    """

    def __init__(self, image_data, transform=None):
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_bytes = self.image_data.iloc[idx, 0]
        label = self.image_data.iloc[idx, 1]
        decoded_img = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self.transform:
            decoded_img = self.transform(decoded_img)

        return decoded_img, label


# ---------------------------- Data Loading Function -----------------------------
def load_data(parquet_file_path, batch_size):
    """
    Load dataset from a parquet file, split into train/test, and create DataLoaders.

    Args:
        parquet_file_path (str): Path to the parquet file.
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
        num_classes (int): Number of unique classes in the dataset.
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    # Load data from parquet file
    data = pd.read_parquet(parquet_file_path)

    # Split into training and testing datasets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create Dataset objects
    train_dataset = ImageByteDataset(train_data, transform=transform)
    test_dataset = ImageByteDataset(test_data, transform=transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine the number of unique classes
    num_classes = len(data.iloc[:, 1].unique())

    return train_loader, test_loader, num_classes


# ---------------------------- Model Creation Function ---------------------------

def create_model(num_classes):
    """
    Create and configure a Swin Transformer model for fine-tuning.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        peft_model (torch.nn.Module): Model with LoRA PEFT configuration applied.
    """

    # Load pretrained Swin Transformer with updated weights argument
    weights = Swin_T_Weights.IMAGENET1K_V1  # Specify the desired pretrained weights
    base_model = swin_t(weights=weights)  # Use weights argument instead of pretrained
    base_model.head = torch.nn.Linear(768, num_classes)

    # Configure LoRA for PEFT
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["qkv", "proj", "head"],
        lora_dropout=0.1,
        inference_mode=False
    )

    # Apply PEFT to the model
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()
    peft_model.to(device)

    return peft_model


# ----------------------------- Training Function --------------------------------
def train_model(peft_model, num_epochs, learning_rate, train_loader):
    """
    Train the PEFT model using the provided training DataLoader.

    Args:
        peft_model (torch.nn.Module): The PEFT-configured model.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        train_loader (DataLoader): DataLoader for the training set.

    Returns:
        peft_model (torch.nn.Module): Trained model.
    """
    # Define optimizer and loss function
    optimizer = AdamW(peft_model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        peft_model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients, forward pass, and backward pass
            optimizer.zero_grad()
            outputs = peft_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute epoch metrics
        epoch_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}")

    return peft_model


# ----------------------------- Evaluation Function ------------------------------
def eval_model(peft_model, output_dir, test_loader):
    """
    Evaluate the PEFT model on the test set and save the trained model.

    Args:
        peft_model (torch.nn.Module): Trained PEFT model.
        output_dir (str): Directory to save the model.
        test_loader (DataLoader): DataLoader for the test set.
    """
    peft_model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = peft_model(inputs)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}")

    # Save the model
    torch.save(peft_model.state_dict(), output_dir)


def parse_args():

    parser = argparse.ArgumentParser(description="Swin Transformer Training with PEFT")
    parser.add_argument("--parquet_file_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer.")
    return parser.parse_args()


# ---------------------------------- Main Function ------------------------------
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load data
    train_loader, test_loader, num_classes = load_data(args.parquet_file_path, args.batch_size)

    # Create model
    peft_model = create_model(num_classes)

    # Train model
    peft_model = train_model(peft_model, args.num_epochs, args.learning_rate, train_loader)

    # Evaluate model
    eval_model(peft_model, args.output_dir, test_loader)


if __name__ == "__main__":
    main()
