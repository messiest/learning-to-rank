# Models Directory

This directory stores the serialized model weights, architecture configurations, and tuning logs.

## 📦 Model Formats

* **`.keras`**: The native Keras 3 format. Use this for standard model saving and loading within the library.
* **`.h5`**: Legacy Keras format. Supported by `EnsembleModel` for backward compatibility.
* **SavedModel/**: Standard TensorFlow directories. These are typically generated for deployment via TensorFlow Serving (TFS) or Vertex AI.

## 🚀 Best Practices

1.  **Naming Convention**: It is recommended to include the architecture type and a version or timestamp (e.g., `transformer_v1_20260202.keras`).
2.  **Ensembling**: When using `EnsembleModel`, point the initializer to a list of paths within this directory.
3.  **Tuning**: The `keras_tuner` output logs are usually stored here in a sub-directory (e.g., `models/tuning_search/`) to keep track of the best hyperparameter trials.
