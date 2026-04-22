"""
5×5 repeated stratified k-fold cross-validation for DNN and transfer
learning models, mirroring pred_dnn_w_transfer_learning.py.

Replaces the 10-iteration random-split loop with a proper
RepeatedStratifiedKFold(n_splits=5, n_repeats=5), giving 25 folds total.
Stratification preserves the active/inactive ratio in every fold, which
matters for the imbalanced M1 dataset.

Early-stopping validation set:
  A 10% inner split is drawn from each fold's *training* portion only.
  The held-out fold is never seen during training — it is used solely
  for evaluation, so there is no leakage.

Outputs (in ../Data/Results/M1/):
  dnn_REINVENT4_5x5cv_folds.csv   — per-fold metrics (25 rows)
  dnn_REINVENT4_5x5cv_summary.csv — mean ± std across all 25 folds
  dnn_REINVENT4_5x5cv_test.csv    — scaffold-test performance per fold
  tl_5x5cv_folds.csv
  tl_5x5cv_summary.csv
  tl_5x5cv_test.csv

Usage:
    cd Scripts/
    python pred_dnn_5x5cv.py
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, callbacks
from tensorflow.keras.layers import Dense, Dropout

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_FP_PATH = "../Data/Training/M1/M1_training_set_REINVENT4_FPs.csv"
TEST_FP_PATH  = "../Data/Test/M1_test_scaffold_split_FPs.csv"
TL_FP_PATH    = "../Data/Training/M1/TL_GLASS_FPs.csv"
RESULTS_DIR   = "../Data/Results/M1"
TL_BASE_MODEL = "../Model/TL_model_5x5cv.keras"

N_SPLITS  = 5
N_REPEATS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Keras metrics (same set as original script)
# ---------------------------------------------------------------------------
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.AUC(curve="ROC", name="auc"),
]

# ---------------------------------------------------------------------------
# Helpers (unchanged from pred_dnn_w_transfer_learning.py)
# ---------------------------------------------------------------------------

def metrics_calc(data):
    data = data.astype(float)
    data["Sensitivity"] = data["TP"] / (data["TP"] + data["FN"])
    data["Specificity"] = data["TN"] / (data["TN"] + data["FP"])
    data["MCC"] = (
        (data["TP"] * data["TN"]) - (data["FP"] * data["FN"])
    ) / np.sqrt(
        (data["TP"] + data["FP"])
        * (data["TP"] + data["FN"])
        * (data["TN"] + data["FP"])
        * (data["TN"] + data["FN"])
    )
    data["ROC-AUC"] = data["AUC"]
    data["G-Mean"] = np.sqrt(data["Sensitivity"] * data["Specificity"])
    return data


def build_dnn(input_dim, dropout=0.25, lr=0.001, n_hidden1=1000, n_hidden2=500):
    model = Sequential([
        layers.Dense(n_hidden1, activation="relu", input_shape=(input_dim,)),
        layers.Dropout(dropout),
        layers.Dense(n_hidden2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=METRICS,
    )
    return model


def run_dnn(X_train, y_train, X_val, y_val, model_path, epochs=2000, batch_size=64, patience=50):
    model = build_dnn(X_train.shape[1])
    cb_list = [
        callbacks.ModelCheckpoint(model_path, save_best_only=True),
        callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb_list, verbose=0,
    )
    return model


def run_transfer_learning(
    base_model_path, X_train, y_train, X_val, y_val, output_path,
    epochs=2000, batch_size=64, patience=50,
):
    base_model = tf.keras.models.load_model(base_model_path)
    model = Sequential(base_model.layers)
    for layer in model.layers[:-3]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=METRICS,
    )
    cb_list = [
        callbacks.ModelCheckpoint(output_path, save_best_only=True),
        callbacks.EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb_list, verbose=0,
    )
    return model

# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

EVAL_COLS   = ["Loss", "TP", "FP", "TN", "FN", "AUC"]
REPORT_COLS = ["Sensitivity", "Specificity", "MCC", "ROC-AUC", "G-Mean"]


def save_fold_results(rows, csv_path):
    """Save per-fold metrics and return the DataFrame."""
    col_names = ["Repeat", "Fold"] + EVAL_COLS
    df = pd.DataFrame(rows, columns=col_names)
    df = metrics_calc(df)
    df[["Repeat", "Fold"] + REPORT_COLS].to_csv(csv_path, index=False)
    logger.info("Fold results saved to %s", csv_path)
    return df


def save_summary(df, csv_path):
    """Save mean ± std summary across all folds."""
    summary = df[REPORT_COLS].agg(["mean", "std"]).T
    summary.columns = ["Mean", "Std"]
    summary.to_csv(csv_path)
    logger.info("Summary saved to %s", csv_path)
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading training fingerprints: %s", TRAIN_FP_PATH)
    train_fp = pd.read_csv(TRAIN_FP_PATH, index_col=0)
    y_full = np.asarray(train_fp.pop("Activity")).ravel()       # 1-D for KFold
    X_full = train_fp.to_numpy()

    logger.info("Loading test fingerprints: %s", TEST_FP_PATH)
    test_fp = pd.read_csv(TEST_FP_PATH, index_col=0)
    y_test = np.asarray(test_fp.pop("Activity")).reshape(-1, 1)
    X_test = test_fp.to_numpy()

    logger.info(
        "Dataset: %d training samples (%d active, %d inactive), %d test samples",
        len(y_full), y_full.sum(), (y_full == 0).sum(), len(y_test),
    )

    # ------------------------------------------------------------------
    # Pre-train TL base model once on GLASS data (fixed split, seed=42)
    # The GLASS set is independent of M1, so it sits outside the CV loop.
    # ------------------------------------------------------------------
    run_tl = os.path.exists(TL_FP_PATH)
    if run_tl:
        logger.info("Pre-training TL base model on GLASS data ...")
        tl_fp = pd.read_csv(TL_FP_PATH, index_col=0)
        y_tl = np.asarray(tl_fp.pop("Activity")).reshape(-1, 1)
        X_tl = tl_fp.to_numpy()
        X_tl_tr, X_tl_val, y_tl_tr, y_tl_val = train_test_split(
            X_tl, y_tl, test_size=0.1, random_state=RANDOM_STATE, shuffle=True
        )
        run_dnn(X_tl_tr, y_tl_tr, X_tl_val, y_tl_val, TL_BASE_MODEL)
        logger.info("TL base model saved to %s", TL_BASE_MODEL)
    else:
        logger.warning("TL_GLASS_FPs.csv not found — skipping TL model.")

    # ------------------------------------------------------------------
    # 5×5 repeated stratified k-fold CV
    # ------------------------------------------------------------------
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    dnn_cv_rows, dnn_test_rows = [], []
    tl_cv_rows,  tl_test_rows  = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_full, y_full)):
        repeat = fold_idx // N_SPLITS + 1
        fold   = fold_idx  % N_SPLITS + 1
        logger.info("--- Repeat %d/%d | Fold %d/%d ---", repeat, N_REPEATS, fold, N_SPLITS)

        X_fold_train = X_full[train_idx]
        X_fold_val   = X_full[val_idx]
        y_fold_train = y_full[train_idx].reshape(-1, 1)
        y_fold_val   = y_full[val_idx].reshape(-1, 1)

        # Inner 10% split from the training fold for early-stopping only.
        # The held-out fold (X_fold_val) is never touched during training.
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_fold_train, y_fold_train, test_size=0.1, shuffle=True, random_state=None
        )

        # Use a temp file so 25 model files don't accumulate on disk
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # ---- DNN ----
            model_dnn = run_dnn(X_tr, y_tr, X_es, y_es, tmp_path)
            dnn_cv_rows.append(
                [repeat, fold] + model_dnn.evaluate(X_fold_val, y_fold_val, verbose=0)
            )
            dnn_test_rows.append(
                [repeat, fold] + model_dnn.evaluate(X_test, y_test, verbose=0)
            )
            logger.info(
                "  DNN CV  — loss %.4f | AUC %.4f",
                dnn_cv_rows[-1][2], dnn_cv_rows[-1][-1],
            )

            # ---- Transfer learning ----
            if run_tl:
                model_tl = run_transfer_learning(
                    TL_BASE_MODEL, X_tr, y_tr, X_es, y_es, tmp_path
                )
                tl_cv_rows.append(
                    [repeat, fold] + model_tl.evaluate(X_fold_val, y_fold_val, verbose=0)
                )
                tl_test_rows.append(
                    [repeat, fold] + model_tl.evaluate(X_test, y_test, verbose=0)
                )
                logger.info(
                    "  TL  CV  — loss %.4f | AUC %.4f",
                    tl_cv_rows[-1][2], tl_cv_rows[-1][-1],
                )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    dnn_cv_df   = save_fold_results(dnn_cv_rows,   f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_folds.csv")
    dnn_test_df = save_fold_results(dnn_test_rows,  f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_test.csv")
    dnn_cv_sum  = save_summary(dnn_cv_df,           f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_summary.csv")
    _           = save_summary(dnn_test_df,         f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_test_summary.csv")

    logger.info("\nDNN CV performance (mean ± std across %d folds):", N_SPLITS * N_REPEATS)
    logger.info("\n%s", dnn_cv_sum.to_string())

    if run_tl:
        tl_cv_df   = save_fold_results(tl_cv_rows,   f"{RESULTS_DIR}/tl_5x5cv_folds.csv")
        tl_test_df = save_fold_results(tl_test_rows,  f"{RESULTS_DIR}/tl_5x5cv_test.csv")
        tl_cv_sum  = save_summary(tl_cv_df,           f"{RESULTS_DIR}/tl_5x5cv_summary.csv")
        _          = save_summary(tl_test_df,         f"{RESULTS_DIR}/tl_5x5cv_test_summary.csv")

        logger.info("\nTL CV performance (mean ± std across %d folds):", N_SPLITS * N_REPEATS)
        logger.info("\n%s", tl_cv_sum.to_string())


if __name__ == "__main__":
    main()
