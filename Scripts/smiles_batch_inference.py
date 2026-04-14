#!/usr/bin/env python3
"""
Batch SMILES inference using DNN_REINVENT4.keras.

Converts SMILES to 1024-bit Morgan fingerprints (radius=2) — matching the
training pipeline — and runs them through DNN_REINVENT4.keras to predict
active (1) or inactive (0) for the M1 muscarinic receptor.

Designed for large inputs (1M+ SMILES): processes in configurable chunks to
keep memory under control, and featurizes in parallel across CPU workers.

Output CSV columns:
  SMILES            - original input SMILES
  Canonical_SMILES  - RDKit-canonical (or ChEMBL-standardized) SMILES
  Valid             - whether the SMILES could be featurized
  Probability       - model sigmoid output (higher = more likely active)
  Prediction        - 1 (Active) or 0 (Inactive); -1 for invalid SMILES
  Label             - "Active", "Inactive", or "Invalid"

Usage examples:
  # CSV input with a SMILES column:
  python smiles_batch_inference.py --input compounds.csv --output predictions.csv

  # Plain text input (one SMILES per line):
  python smiles_batch_inference.py --input smiles.txt --output predictions.csv

  # Full options:
  python smiles_batch_inference.py \\
      --input compounds.csv \\
      --output predictions.csv \\
      --model ../Model/DNN_REINVENT4.keras \\
      --smiles_col SMILES \\
      --threshold 0.5 \\
      --batch_size 4096 \\
      --chunk_size 50000 \\
      --workers 8 \\
      --standardize
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Fingerprint parameters — must match training (Morgan radius=2, 1024 bits)
_NBITS = 1024
_DEPTH = 2


# ---------------------------------------------------------------------------
# Featurization workers (module-level so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------

def _compute_fp(smiles: str):
    """Compute 1024-bit Morgan FP from a raw SMILES string.

    Returns (canonical_smiles, fp_array) on success, or (smiles, None) on
    any failure (unparseable SMILES, empty molecule, etc.).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        canonical = Chem.MolToSmiles(mol)
        a = np.zeros(_NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, _DEPTH, _NBITS), a
        )
        return canonical, a
    except Exception:
        return smiles, None


def _standardize_and_compute_fp(smiles: str):
    """ChEMBL-standardize a SMILES string, then compute 1024-bit Morgan FP.

    Mirrors the curate_data.py / add_gen.py preprocessing:
      1. Parse with RDKit
      2. Reject exclude-flagged structures (mixtures, inorganics, etc.)
      3. Strip salts via chembl_structure_pipeline
      4. Standardize via chembl_structure_pipeline
      5. Compute Morgan FP on the standardized molecule

    Returns (canonical_smiles, fp_array) on success, or (smiles, None) on
    any failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from chembl_structure_pipeline import standardizer, exclude_flag

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, None
        if exclude_flag.exclude_flag(mol):
            return smiles, None

        parent, _ = standardizer.get_parent_mol(mol)
        std_mol = standardizer.standardize_mol(parent)
        canonical = Chem.MolToSmiles(std_mol)

        mol2 = Chem.MolFromSmiles(canonical)
        if mol2 is None:
            return smiles, None

        a = np.zeros(_NBITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol2, _DEPTH, _NBITS), a
        )
        return canonical, a
    except Exception:
        return smiles, None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_smiles(input_path: str, smiles_col: str = None):
    """Load SMILES from a CSV/TSV (with header) or a plain-text file.

    For CSV/TSV: returns the DataFrame and the resolved SMILES column name.
    For plain text: wraps lines into a DataFrame with a 'SMILES' column.

    Args:
        input_path: Path to the input file.
        smiles_col: Column name to use for SMILES. Auto-detected if None.

    Returns:
        (df, smiles_col_name)
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(input_path, sep=sep)
        if smiles_col is None:
            candidates = [c for c in df.columns if "smiles" in c.lower()]
            smiles_col = candidates[0] if candidates else df.columns[0]
            logger.info("Auto-detected SMILES column: %r", smiles_col)
        if smiles_col not in df.columns:
            raise ValueError(
                f"Column {smiles_col!r} not found. Available: {list(df.columns)}"
            )
        return df, smiles_col
    else:
        # Plain text: one SMILES per line, ignore blank lines and comments
        with open(input_path) as fh:
            smiles = [
                ln.strip()
                for ln in fh
                if ln.strip() and not ln.startswith("#")
            ]
        logger.info("Read %d SMILES from plain-text file.", len(smiles))
        return pd.DataFrame({"SMILES": smiles}), "SMILES"


# ---------------------------------------------------------------------------
# Chunked featurization + inference
# ---------------------------------------------------------------------------

def featurize_chunk(smiles_list, worker_fn, n_workers: int):
    """Featurize a list of SMILES in parallel using ProcessPoolExecutor.

    Args:
        smiles_list: List of SMILES strings.
        worker_fn:   Module-level function (_compute_fp or
                     _standardize_and_compute_fp).
        n_workers:   Number of parallel processes.

    Returns:
        List of (canonical_smiles, fp_array_or_None) tuples, same length and
        order as smiles_list.
    """
    if n_workers <= 1:
        return [worker_fn(s) for s in smiles_list]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker_fn, smiles_list, chunksize=256))
    return results


def run_inference_chunk(model, smiles_list, feat_results, threshold: float, batch_size: int):
    """Run model inference on featurized results for one chunk.

    Args:
        model:        Loaded Keras model.
        smiles_list:  Original SMILES strings for this chunk.
        feat_results: List of (canonical, fp_or_None) from featurize_chunk.
        threshold:    Probability cutoff for Active classification.
        batch_size:   Keras predict batch size.

    Returns:
        pd.DataFrame with columns: SMILES, Canonical_SMILES, Valid,
        Probability, Prediction, Label.
    """
    canonical_list = []
    valid_mask = []
    fp_list = []
    valid_indices = []  # positions in chunk that have a valid FP

    for i, (canon, fp) in enumerate(feat_results):
        if fp is None:
            valid_mask.append(False)
            canonical_list.append(None)
        else:
            valid_mask.append(True)
            canonical_list.append(canon)
            fp_list.append(fp)
            valid_indices.append(i)

    probs = np.full(len(smiles_list), np.nan, dtype=np.float32)
    preds = np.full(len(smiles_list), -1, dtype=np.int8)

    if fp_list:
        X = np.vstack(fp_list)  # shape (n_valid, 1024)
        raw_probs = model.predict(X, batch_size=batch_size, verbose=0).flatten()
        for arr_i, chunk_i in enumerate(valid_indices):
            p = float(raw_probs[arr_i])
            probs[chunk_i] = p
            preds[chunk_i] = 1 if p >= threshold else 0

    labels = []
    for v, pred in zip(valid_mask, preds):
        if not v:
            labels.append("Invalid")
        elif pred == 1:
            labels.append("Active")
        else:
            labels.append("Inactive")

    return pd.DataFrame(
        {
            "SMILES": smiles_list,
            "Canonical_SMILES": canonical_list,
            "Valid": valid_mask,
            "Probability": probs,
            "Prediction": preds,
            "Label": labels,
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with DNN_REINVENT4.keras on SMILES input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Input file: CSV/TSV with a SMILES column, or plain text (one SMILES per line).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path for predictions.",
    )
    parser.add_argument(
        "--model", default=os.path.join(os.path.dirname(__file__), "..", "Model", "DNN_REINVENT4.keras"),
        help="Path to the Keras model file.",
    )
    parser.add_argument(
        "--smiles_col", default=None,
        help="SMILES column name in CSV input. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for Active classification.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096,
        help="Keras model.predict() batch size.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=50_000,
        help="Number of SMILES processed per iteration (memory control).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel CPU workers for featurization.",
    )
    parser.add_argument(
        "--standardize", action="store_true",
        help=(
            "Apply ChEMBL structure standardization before featurizing "
            "(matches training pipeline; slower for large inputs). "
            "By default, only RDKit canonicalization is applied."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    import tensorflow as tf

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info("Model input shape: %s", model.input_shape)

    # ------------------------------------------------------------------
    # Load SMILES
    # ------------------------------------------------------------------
    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Loading SMILES from: %s", args.input)
    df, smiles_col = load_smiles(args.input, args.smiles_col)
    total = len(df)
    logger.info("Total SMILES to process: %d", total)

    worker_fn = _standardize_and_compute_fp if args.standardize else _compute_fp
    mode_label = "ChEMBL-standardized" if args.standardize else "RDKit-canonical"
    logger.info("Featurization mode: %s | Workers: %d | Chunk size: %d",
                mode_label, args.workers, args.chunk_size)

    # ------------------------------------------------------------------
    # Chunked processing
    # ------------------------------------------------------------------
    n_chunks = max(1, (total + args.chunk_size - 1) // args.chunk_size)
    all_chunks = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, total)
        chunk_smiles = df[smiles_col].iloc[start:end].tolist()

        logger.info(
            "Chunk %d/%d — rows %d–%d (%d SMILES)",
            chunk_idx + 1, n_chunks, start + 1, end, len(chunk_smiles),
        )

        feat_results = featurize_chunk(chunk_smiles, worker_fn, args.workers)
        chunk_df = run_inference_chunk(
            model, chunk_smiles, feat_results, args.threshold, args.batch_size
        )
        all_chunks.append(chunk_df)

        n_valid = chunk_df["Valid"].sum()
        n_active = (chunk_df["Prediction"] == 1).sum()
        n_inactive = (chunk_df["Prediction"] == 0).sum()
        logger.info(
            "  Valid: %d/%d | Active: %d | Inactive: %d | Invalid: %d",
            n_valid, len(chunk_smiles), n_active, n_inactive,
            len(chunk_smiles) - n_valid,
        )

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    result_df = pd.concat(all_chunks, ignore_index=True)

    # Preserve any extra columns from the original CSV (e.g., IDs)
    extra_cols = [c for c in df.columns if c != smiles_col]
    if extra_cols:
        result_df = pd.concat(
            [df[extra_cols].reset_index(drop=True), result_df], axis=1
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_valid = int(result_df["Valid"].sum())
    n_invalid = total - n_valid
    n_active = int((result_df["Prediction"] == 1).sum())
    n_inactive = int((result_df["Prediction"] == 0).sum())

    logger.info("=" * 55)
    logger.info("Results saved to: %s", args.output)
    logger.info("%-25s %10d", "Total SMILES:", total)
    logger.info("%-25s %10d  (%.1f%%)", "Valid:", n_valid, 100 * n_valid / total)
    logger.info("%-25s %10d  (%.1f%%)", "Invalid/skipped:", n_invalid, 100 * n_invalid / total)
    if n_valid > 0:
        logger.info("%-25s %10d  (%.1f%% of valid)", "Predicted Active:", n_active, 100 * n_active / n_valid)
        logger.info("%-25s %10d  (%.1f%% of valid)", "Predicted Inactive:", n_inactive, 100 * n_inactive / n_valid)
    logger.info("Threshold used: %.2f", args.threshold)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
