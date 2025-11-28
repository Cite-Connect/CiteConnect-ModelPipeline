# scripts/model_bias_slicing.py

from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# ---------- Paths ----------
RESULTS_PATH = Path("offline_evaluation_results.json")
METADATA_PATH = Path("data/combined_gcs_data.parquet")  # adjust if your path is different

BIAS_REPORT_PATH = Path("model_bias_report.json")
FAIRNESS_CONFIG_PATH = Path("fairness_config.json")
PLOT_DIR = Path("bias_plots")
PLOT_DIR.mkdir(exist_ok=True)


def extract_primary_field(fields):
    """
    Take the first field from fieldsOfStudy list.
    If missing / empty, return 'Unknown'.
    """
    if isinstance(fields, list) and len(fields) > 0:
        return fields[0]
    return "Unknown"


def run_model_bias_analysis(
    results_path: Path = RESULTS_PATH,
    metadata_path: Path = METADATA_PATH,
    report_path: Path = BIAS_REPORT_PATH,
    fairness_config_path: Path = FAIRNESS_CONFIG_PATH,
):
    # ---------------------------------------------------
    # 1) Load offline evaluation results
    # ---------------------------------------------------
    if not results_path.exists():
        raise FileNotFoundError(f"Offline evaluation results not found at {results_path}")

    with results_path.open() as f:
        results = json.load(f)

    # results["all_results"] is a list of dicts with paper_id, precision_at_k, recall_at_k, etc.
    all_results = results.get("all_results", [])
    if not all_results:
        raise ValueError("No entries found in offline_evaluation_results.json['all_results'].")

    results_df = pd.DataFrame(all_results)

    if "paper_id" not in results_df.columns:
        raise ValueError("Expected 'paper_id' column in offline evaluation results.")

    # ---------------------------------------------------
    # 2) Load metadata with fields of study / year
    # ---------------------------------------------------
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found at {metadata_path}")

    meta_df = pd.read_parquet(metadata_path)

    # Align ID column name
    if "paperId" in meta_df.columns:
        meta_df = meta_df.rename(columns={"paperId": "paper_id"})

    if "paper_id" not in meta_df.columns:
        raise ValueError("Metadata must contain 'paperId' or 'paper_id' column.")

    # Create a primary_field column from fieldsOfStudy
    if "fieldsOfStudy" in meta_df.columns:
        meta_df["primary_field"] = meta_df["fieldsOfStudy"].apply(extract_primary_field)
    else:
        meta_df["primary_field"] = "Unknown"

    # Optional: keep year / citationCount for debugging
    for col in ["year", "citationCount"]:
        if col not in meta_df.columns:
            meta_df[col] = None

    meta_df = meta_df[["paper_id", "primary_field", "year", "citationCount"]]

    # ---------------------------------------------------
    # 3) Join eval results with metadata
    # ---------------------------------------------------
    merged = results_df.merge(meta_df, on="paper_id", how="left")

    # Some rows may have no metadata (paper not in parquet) – that's ok
    # but we'll still see them as primary_field='Unknown'
    merged["primary_field"] = merged["primary_field"].fillna("Unknown")

    # ---------------------------------------------------
    # 4) Compute per-slice metrics by primary_field
    # ---------------------------------------------------
    # We assume columns like precision_at_k, recall_at_k, mrr exist in results
    grouped = merged.groupby("primary_field").agg(
        mean_precision_at_10=("precision_at_k", "mean"),
        mean_recall_at_10=("recall_at_k", "mean"),
        mean_mrr=("mrr", "mean"),
        num_papers=("paper_id", "count"),
    ).reset_index()

    # ---------------------------------------------------
    # 5) Compute fairness disparity on slices with enough support
    # ---------------------------------------------------
    # To avoid noise, only consider slices with at least N papers
    MIN_SUPPORT = 5
    g = grouped[grouped["num_papers"] >= MIN_SUPPORT].copy()

    fairness_summary = {
        "note": f"Not enough slices with >= {MIN_SUPPORT} papers to compute disparities reliably."
    }
    under_served_fields = []

    if not g.empty:
        # Best & worst by mean_precision_at_10
        max_idx = g["mean_precision_at_10"].idxmax()
        min_idx = g["mean_precision_at_10"].idxmin()

        max_row = g.loc[max_idx]
        min_row = g.loc[min_idx]

        max_p = float(max_row["mean_precision_at_10"])
        min_p = float(min_row["mean_precision_at_10"])

        ratio = float(min_p / max_p) if max_p > 0 else None
        diff = float(max_p - min_p)

        fairness_summary = {
            "min_support": MIN_SUPPORT,
            "best_field": max_row["primary_field"],
            "worst_field": min_row["primary_field"],
            "best_precision_at_10": max_p,
            "worst_precision_at_10": min_p,
            "precision_ratio_min_over_max": ratio,
            "precision_diff_max_minus_min": diff,
        }

        # Define "under-served" as slices with precision < 0.8 * best
        RATIO_THRESHOLD = 0.8
        threshold = RATIO_THRESHOLD * max_p

        under_served_fields = g.loc[
            g["mean_precision_at_10"] < threshold, "primary_field"
        ].tolist()

    # ---------------------------------------------------
    # 6) Save full bias report JSON
    # ---------------------------------------------------
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_records_evaluated": int(len(merged)),
        "num_slices": int(grouped["primary_field"].nunique()),
        "per_slice_metrics": grouped.to_dict(orient="records"),
        "fairness_summary": fairness_summary,
    }

    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Model bias report written to {report_path}")

    # ---------------------------------------------------
    # 7) Save a small fairness_config.json
    # ---------------------------------------------------
    fairness_config = {
        "generated_at": datetime.utcnow().isoformat(),
        "metric": "precision_at_10",
        "min_support": MIN_SUPPORT,
        "ratio_threshold": 0.8,
        "under_served_fields": under_served_fields,
    }

    with fairness_config_path.open("w") as f:
        json.dump(fairness_config, f, indent=2)

    print(f"✅ Fairness config written to {fairness_config_path}")
    print("   Under-served fields:", under_served_fields)

    # ---------------------------------------------------
    # 8) Optional: bar plot of mean_precision_at_10 by field
    # ---------------------------------------------------
    if not grouped.empty:
        plt.figure(figsize=(10, 6))
        grouped_sorted = grouped.sort_values("mean_precision_at_10", ascending=False)
        plt.bar(grouped_sorted["primary_field"], grouped_sorted["mean_precision_at_10"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean Precision@10")
        plt.title("Model performance by primary field (Precision@10)")
        plt.tight_layout()
        plot_path = PLOT_DIR / "model_precision_by_field.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Plot saved to {plot_path}")


if __name__ == "__main__":
    run_model_bias_analysis()
