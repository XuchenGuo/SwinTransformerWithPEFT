# Swin Transformer with PEFT (LoRA)

This is an implementation for fine-tuning Swin Transformer models using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)**.

---

## Example Training Results

Below are sample results from training the Swin Transformer with PEFT:

### Epoch Metrics
- **Epoch 1/10**  
  - Loss: `0.7421`  
  - Accuracy: `81.23%`  
  - Precision: `81.05%`  

- **Epoch 2/10**  
  - Loss: `0.5217`  
  - Accuracy: `89.45%`  
  - Precision: `89.30%`  

### Test Results
- **Test Accuracy**: `92.13%`  
- **Test Precision**: `91.87%`  

---
## How It Works

### Dataset Preparation
The script expects the dataset to be provided as a Parquet file with the following structure:
- **`image_bytes`**: The raw image data stored as bytes.
- **`label`**: The corresponding classification label.

### Transformations
To prepare the data for training, the following transformations are applied:
1. **Resizing**: Images are resized to `224x224` pixels.
2. **Normalization**: Pixel values are normalized using ImageNet's mean and standard deviation values.

### Fine-Tuning
The model is fine-tuned using the following steps:
1. **Model Initialization**: A pretrained **Swin Transformer (`swin_t`)** is loaded with ImageNet weights.
2. **LoRA PEFT**: Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) optimizes specific layers, including:
   - `qkv` (query, key, value layers)
   - `proj` (projection layers)
   - `head` (classification head)

### Key Features:
- **Data Handling**: Reads image data from Parquet files and decodes them from byte format.
- **Model Fine-Tuning**: Adapts a pretrained Swin Transformer using LoRA PEFT for efficient parameter updates.
- **Performance Evaluation**: Computes key metrics such as accuracy and precision for training and testing phases.
- **Embeddings Extraction**: Saves the trained model and extracts embeddings for test datasets, and feed them into df-analyze.

---

## Installation

Follow these steps to install the required dependencies and set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/XuchenGuo/SwinTransformerWithPEFT.git
   cd SwinTransformerWithPEFT

   pip install -r requirements.txt

## Start

Run the script with the following command:

```bash
python image_10.py --parquet_file_path /path/to/dataset.parquet \
                   --output_dir /path/to/save/model \
                   --num_epochs 10 \
                   --batch_size 32 \
                   --learning_rate 0.01
