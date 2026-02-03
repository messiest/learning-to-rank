# Learning to Rank (LTR) with TensorFlow

This project implements a modular Learning to Rank pipeline using the MSLR-WEB10K and MSLR-WEB30K datasets. It supports multiple architectures, including ListNet, Residual Rankers, and Transformer-based Context-Aware models.

## 🚀 Quick Start

### 1. Installation

This project uses `uv` for lightning-fast dependency management.

```bash
# Install dependencies
uv sync

```

### 2. Data Preparation

Download the datasets and convert them to `SequenceExample` TFRecords for optimal performance.

```bash
# Download MSLR-WEB10K
python scripts/download_data.py

# Convert LIBSVM text to TFRecord
python scripts/convert_data.py --input_dir data/raw/MSLR-WEB10K --output_dir data/processed/MSLR-WEB10K

```

### 3. Training & Hyperparameter Tuning

The unified training script allows you to tune different architectures across all 5 folds.

```bash
# Tune the Transformer model on Fold 1
python scripts/train_and_tune.py --model_type transformer --single_fold Fold1

# Tune the ResListNet model on all folds
python scripts/train_and_tune.py --model_type reslistnet

```

### 4. Export the Ensemble

Combine your best-performing models into a single computation graph for production.

```bash
python scripts/export_ensemble.py \
    models/tuned/transformer_Fold1_best.keras \
    models/tuned/reslistnet_Fold2_best.keras \
    --export_path serving_model/ensemble/1

```

---

## 🛠 Project Architecture

| Component | Responsibility |
| --- | --- |
| **`data.py`** | Handles `SequenceExample` parsing and query-level `padded_batch`. |
| **`models/`** | Implements Pointwise (ListNet) and Listwise (Transformer) rankers. |
| **`losses.py`** | Listwise Softmax Cross-Entropy with numerical stability logic. |
| **`metrics.py`** | GPU-accelerated NDCG calculation using `top_k` and `gather_nd`. |
| **`ensemble.py`** | A `tf.Module` wrapper that averages predictions across sub-models. |

---

## 🐳 Production Serving

We use **TensorFlow Serving** to expose the ensemble via REST and gRPC.

### Start the Server

Ensure you have Docker installed, then run:

```bash
docker-compose up -d

```

### Query the Model (Inference)

The model expects a 3D tensor shape of `[Batch, List_Size, 136]`.

```bash
curl -X POST http://localhost:8501/v1/models/ltr_ensemble:predict \
    -d '{
      "instances": [
        [[0.5, 0.1, ...], [0.8, 0.2, ...], [0.1, 0.9, ...]]
      ]
    }'

```

---

## 📈 Performance Notes

* **Mixed Precision:** Training is set to `mixed_float16` to leverage Tensor Cores on NVIDIA GPUs.
* **Context-Awareness:** The Transformer model uses self-attention to allow documents to "compete" within the query group, often resulting in higher NDCG scores compared to standard ListNet.

---

This project is now a complete end-to-end LTR system. **Is there anything else you’d like to add, such as a script for "Distilling" the ensemble into a smaller, faster model for low-latency ranking?**