# Data Directory

This directory contains the raw and processed datasets used for training and evaluating the ranking models.

## 📊 Dataset References

This project is primarily designed for the **Microsoft Learning to Rank (MSLR-WEB)** datasets. These datasets consist of query-document pairs with relevance labels ranging from 0 (irrelevant) to 4 (perfectly relevant).

* **[MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/):** Contains 10,000 queries. Each row represents a document with 136 features.
* **[MSLR-WEB30K](https://www.microsoft.com/en-us/research/project/mslr/):** A larger set containing 30,000 queries, following the same feature format as WEB10K.

## 📂 Structure

* `raw/`: Store the original `.txt` files (e.g., `train.txt`, `test.txt`) here.
* `tfrecords/`: Output directory for the `ltr-convert` script. Contains `SequenceExample` serialized data for high-performance training.

## ⚙️ Conversion

To transform raw LIBSVM text files into optimized TFRecords, use the provided script:

```bash
ltr-convert --input_files data/raw/train.txt --output_path data/tfrecords/train.tfrecord
```
