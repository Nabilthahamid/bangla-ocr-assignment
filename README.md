# Bangla Handwritten Word OCR

**Name:** MD Nabil Thahamid Chowdhury  
**Student ID:** 1000127525  
**GitHub repo:** https://github.com/Nabilthahamid/bangla-ocr-assignment

A CNN trained on the BanglaLekha-Isolated dataset (84 character classes), wrapped in a Streamlit app where you draw one Bangla word and the app predicts it. Training is tracked with MLflow.

## Approach

1. **Dataset.** BanglaLekha-Isolated: 84 classes (`Images/1` ... `Images/84`), one folder per character. The data is split 80% train / 10% validation / 10% test, stratified by class.
2. **Preprocessing** (`preprocess_char_image` in `train.py`): invert to white ink on a black background, pad to a square to keep the aspect ratio, resize to 64x64, and scale pixel values to 0-1.
3. **Models.** Two architectures, selected with `--arch`:
   - `baseline_cnn`: 3 conv + max-pool blocks, dense layer, dropout.
   - `deep_bn_cnn`: 4 conv blocks with batch normalization, global average pooling, dense layer, dropout.
4. **Experiment tracking.** Each run of `train.py` logs parameters, per-epoch train/validation loss and accuracy, final test accuracy and the model to MLflow (`artifacts/mlflow`). The trained model is saved to `models/model.keras` and the class-to-character mapping to `labels.json`.
5. **Word recognition** (`app.py`):
   - The drawn word is binarized and cropped.
   - The top horizontal line (matra) is detected and temporarily removed so joined letters separate.
   - Connected components are extracted left to right, and each one is classified by the CNN (top-5 probabilities kept).
   - Because the dataset has no dependent vowel signs (kars), the sequence is matched against a built-in Bangla lexicon using a probability-weighted edit distance, which restores the vowel signs.

## Project structure

```
bangla-ocr-assignment/
|-- train.py
|-- app.py
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- labels.json
|-- models/model.keras
|-- artifacts/mlflow/
`-- screenshots/
    |-- streamlit_app.png
    `-- mlflow_experiment.png
```

## How to run

### 1. Install

```
pip install -r requirements.txt
```

### 2. Train (records the run in MLflow)

Download BanglaLekha-Isolated and point `--data_dir` at the `Images` folder (it must contain subfolders `1` ... `84`).

```
python train.py --data_dir "path/to/BanglaLekha-Isolated/Images" --arch baseline_cnn --epochs 12
python train.py --data_dir "path/to/BanglaLekha-Isolated/Images" --arch deep_bn_cnn --epochs 12
```

Options: `--epochs`, `--batch_size`, `--lr`, `--img_size`, `--arch {baseline_cnn,deep_bn_cnn}`, `--max_per_class` (faster CPU training). `DATA_DIR` can be set as an environment variable instead of `--data_dir`.

### 3. View MLflow

```
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:./artifacts/mlflow
```

Open http://localhost:5000.

### 4. Run the app locally

```
streamlit run app.py
```

Open http://localhost:8501, draw a word, and click **Predict Word**.

### 5. Run with Docker

`models/model.keras` and `labels.json` must exist before building (run `train.py` first).

```
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open http://localhost:8501.

## Limitations

- The model only knows the 84 isolated characters in the dataset. Dependent vowel signs (kars) and many conjuncts are not classes, so they are recovered only through the lexicon.
- The final word is chosen from a small built-in lexicon (about 90 words). Words outside it are replaced by the closest lexicon word, so they are likely to be wrong.
- Segmentation relies on the matra line and connected components. Touching letters, broken strokes, very slanted writing, or characters with parts above and below the line can be split or merged incorrectly.
- The model is trained on isolated characters written on paper, not on mouse-drawn strokes, so accuracy on the canvas is lower than the test accuracy reported in MLflow.
- Only single words are supported, not full sentences or multiple lines.
- Training was done on CPU with a limited number of epochs, with no data augmentation or hyperparameter search.
