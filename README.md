# YouTube AdView Prediction

Predicts the number of ad views a YouTube video will receive based on engagement metrics (views, likes, dislikes, comments, duration, category). Five regression models are benchmarked: Linear Regression, Decision Tree, Random Forest, SVR, and a Keras ANN.

---

## Dataset

The dataset contains YouTube video metadata with the following columns:

| Column | Description |
|---|---|
| `vidid` | Video identifier |
| `adview` | Target — number of ad views |
| `views` | Total video views |
| `likes` | Number of likes |
| `dislikes` | Number of dislikes |
| `comment` | Number of comments |
| `published` | Publish date |
| `duration` | Video duration (ISO 8601 format: PT#H#M#S) |
| `category` | Video category (A–H) |

**Place your data files at:**
```
data/train.csv
data/test.csv
```

---

## Project Structure

```
youtube-adview-prediction/
│
├── data/                  # train.csv and test.csv (git-ignored)
│
├── notebooks/
│   └── youtube_adview_prediction.ipynb   # Interactive walkthrough
│
├── src/
│   ├── config.py          # All paths, constants, and hyperparameters
│   ├── preprocess.py      # Cleaning, duration parsing, encoding, scaling
│   ├── eda.py             # Visualization functions (category, adview, heatmap)
│   ├── train.py           # Train + evaluate all 5 models
│   └── predict.py         # Run inference on test.csv, save predictions
│
├── models/                # Saved model artifacts (git-ignored)
├── reports/               # model_comparison.csv + PredictedAdview.csv
├── requirements.txt
├── .gitignore
└── README.md
```

### `src/` vs notebook

All logic lives in `src/`. The notebook calls `src/` functions with explanations. You can run the entire pipeline from the command line without Jupyter.

---

## How to Run

### Step 1 — Install dependencies

```bash
pip3 install -r requirements.txt
```

### Step 2 — Add data

Place `train.csv` and `test.csv` in the `data/` folder.

### Step 3 — Train all models

```bash
python3 src/train.py
```

Train a specific model only:
```bash
python3 src/train.py --model rf     # Random Forest
python3 src/train.py --model ann    # ANN
python3 src/train.py --model dt     # Decision Tree
python3 src/train.py --model lr     # Linear Regression
python3 src/train.py --model svr    # SVR
```

Results are saved to `reports/model_comparison.csv`.

### Step 4 — Generate predictions on test data

```bash
python3 src/predict.py              # uses Decision Tree (default)
python3 src/predict.py --model ann  # uses ANN
```

Predictions saved to `reports/PredictedAdview.csv`.

### Step 5 — (Optional) Explore in notebook

```bash
jupyter lab notebooks/
```

---

## Preprocessing Steps

1. **Category encoding** — letters A–H mapped to integers 1–8
2. **Sentinel removal** — rows where views/likes/dislikes/comment == `'F'` are dropped
3. **Type conversion** — engagement columns cast to numeric
4. **Duration parsing** — ISO 8601 format (`PT1H23M45S`) converted to total seconds
5. **Label encoding** — `vidid` and `published` columns label-encoded
6. **Outlier filtering** — training rows with `adview > 2,000,000` removed
7. **MinMaxScaler** — all features scaled to [0, 1]; scaler fitted on train, applied to test

---

## Models

| Model | Key settings |
|---|---|
| Linear Regression | Default sklearn |
| Decision Tree | `random_state=42` |
| Random Forest | 200 trees, max_depth=25, min_samples_split=15 |
| SVR | Default RBF kernel |
| ANN | 2 hidden layers × 6 neurons, ReLU, Adam, 100 epochs |

---

## Bugs Fixed from Original

The original Colab script had several issues that are corrected here:

- **`X_test` scaled with `fit_transform` instead of `transform`** — this leaks test data statistics into the scaler, inflating performance. Fixed to `scaler.transform(X_test)`.
- **SVR evaluated against Linear Regression** — `print_error(X_test, y_test, linear_regression)` was called after fitting SVR. Fixed to evaluate SVR correctly.
- **Duplicate preprocessing of test data** — the entire train preprocessing block was copy-pasted for test with minor variable name changes. Refactored into a single `clean()` function.
- **`keras.optimizers.adam_v2`** — referenced the class instead of instantiating it. Fixed to `keras.optimizers.Adam()`.
- **`np.bool` deprecated** — replaced with `bool` (removed in NumPy 1.24+).
- **Duration parsed by re-reading CSV** — the original re-read `train.csv` from disk inside the duration block unnecessarily. Refactored to operate on the in-memory DataFrame.
