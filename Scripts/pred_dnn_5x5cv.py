"""
5×5 repeated stratified k-fold cross-validation for the DNN model.

Replaces the 10-iteration random-split loop with a proper
RepeatedStratifiedKFold(n_splits=5, n_repeats=5), giving 25 folds total.
Stratification preserves the active/inactive ratio in every fold, which
matters for the imbalanced M1 dataset.

Early-stopping validation set:
  A 10% inner split is drawn from each fold's *training* portion only.
  The held-out fold is never seen during training — it is used solely
  for evaluation, so there is no leakage.

Outputs (in ../Data/Results/M1/):
  dnn_REINVENT4_5x5cv_folds.csv   — per-fold CV metrics (25 rows)
  dnn_REINVENT4_5x5cv_summary.csv — mean ± std across all 25 folds
  dnn_REINVENT4_5x5cv_best_test.csv — scaffold-test metrics for the
                                       single best model (selected by
                                       highest MCC on its held-out fold)

Usage:
    cd Scripts/
    python pred_dnn_5x5cv.py
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, callbacks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_FP_PATH = "train_df_FPs.csv"
TEST_FP_PATH  = "test_df_FPs.csv"
RESULTS_DIR   = "."

N_SPLITS     = 5
N_REPEATS    = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Keras metrics
# ---------------------------------------------------------------------------
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.AUC(curve="ROC", name="auc"),
]

# ---------------------------------------------------------------------------
# Model helpers
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


def run_dnn(X_train, y_train, X_val, y_val, model_path, epochs=2000, batch_size=64, patience=50, class_weight=None):
    model = build_dnn(X_train.shape[1])
    cb_list = [
        callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
        callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb_list, class_weight=class_weight, verbose=0,
    )
    return model

# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

EVAL_COLS   = ["Loss", "TP", "FP", "TN", "FN", "AUC"]
REPORT_COLS = ["Sensitivity", "Specificity", "MCC", "ROC-AUC", "G-Mean"]


def save_fold_results(rows, csv_path):
    col_names = ["Repeat", "Fold"] + EVAL_COLS
    df = pd.DataFrame(rows, columns=col_names)
    df = metrics_calc(df)
    df[["Repeat", "Fold"] + REPORT_COLS].to_csv(csv_path, index=False)
    logger.info("Fold results saved to %s", csv_path)
    return df


def save_summary(df, csv_path):
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
    y_full = np.asarray(train_fp.pop("CLASS")).ravel()  # 1-D for KFold
    X_full = train_fp.to_numpy(dtype=np.float32)

    logger.info("Loading test fingerprints: %s", TEST_FP_PATH)
    test_fp = pd.read_csv(TEST_FP_PATH, index_col=0)
    y_test = np.asarray(test_fp.pop("CLASS")).reshape(-1, 1)
    X_test = test_fp.to_numpy(dtype=np.float32)

    logger.info(
        "Dataset: %d training samples (%d active, %d inactive), %d test samples",
        len(y_full), int(y_full.sum()), int((y_full == 0).sum()), len(y_test),
    )

    # ------------------------------------------------------------------
    # 5×5 repeated stratified k-fold CV
    # ------------------------------------------------------------------
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    dnn_cv_rows = []
    best_model, best_mcc, best_fold_label = None, -1.0, ""

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_full, y_full)):
        repeat = fold_idx // N_SPLITS + 1
        fold   = fold_idx  % N_SPLITS + 1
        logger.info("--- Repeat %d/%d | Fold %d/%d ---", repeat, N_REPEATS, fold, N_SPLITS)

        X_fold_train = X_full[train_idx]
        X_fold_val   = X_full[val_idx]
        y_fold_train = y_full[train_idx].reshape(-1, 1)
        y_fold_val   = y_full[val_idx].reshape(-1, 1)

        # Inner 10% split from the training fold for early stopping only.
        # The held-out fold (X_fold_val) is never touched during training.
        # Stratified so the early-stopping val set always contains both classes.
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_fold_train, y_fold_train, test_size=0.1, shuffle=True,
            stratify=y_fold_train, random_state=None
        )

        # Class weights computed from the inner training set so the loss
        # penalises minority-class (active) errors proportionally to the
        # imbalance ratio in this specific fold.
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr.ravel())
        class_weight = {0: weights[0], 1: weights[1]}

        # Temp file so 25 model checkpoints don't accumulate on disk
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model = run_dnn(X_tr, y_tr, X_es, y_es, tmp_path, class_weight=class_weight)
            cv_metrics = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            dnn_cv_rows.append([repeat, fold] + cv_metrics)

            # cv_metrics order: [loss, tp, fp, tn, fn, auc]
            _, tp, fp, tn, fn, _ = cv_metrics
            fold_mcc = (tp * tn - fp * fn) / np.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            )
            logger.info("  CV loss %.4f | CV MCC %.4f", cv_metrics[0], fold_mcc)

            # Keep the best model in memory — evaluated on test set after the loop
            if fold_mcc > best_mcc:
                best_mcc = fold_mcc
                best_model = model
                best_fold_label = f"repeat {repeat}, fold {fold}"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Save CV results
    # ------------------------------------------------------------------
    dnn_cv_df  = save_fold_results(dnn_cv_rows, f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_folds.csv")
    dnn_cv_sum = save_summary(dnn_cv_df,         f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_summary.csv")

    logger.info("\nDNN CV performance (mean ± std across %d folds):", N_SPLITS * N_REPEATS)
    logger.info("\n%s", dnn_cv_sum.to_string())

    # ------------------------------------------------------------------
    # Evaluate best model on scaffold test set (once)
    # ------------------------------------------------------------------
    logger.info(
        "Best model: %s (CV MCC %.4f) — evaluating on scaffold test set ...",
        best_fold_label, best_mcc,
    )
    test_metrics = best_model.evaluate(X_test, y_test, verbose=0)
    test_row = pd.DataFrame([test_metrics], columns=EVAL_COLS)
    test_row = metrics_calc(test_row)
    test_out = pd.concat(
        [pd.DataFrame({"Best_Fold": [best_fold_label]}), test_row[REPORT_COLS]], axis=1
    )
    test_out.to_csv(f"{RESULTS_DIR}/dnn_REINVENT4_5x5cv_best_test.csv", index=False)
    logger.info("Test results saved to %s/dnn_REINVENT4_5x5cv_best_test.csv", RESULTS_DIR)
    logger.info("Test set performance:\n%s", test_out[REPORT_COLS].to_string(index=False))


if __name__ == "__main__":
    main()
