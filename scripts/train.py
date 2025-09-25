# scripts/train.py
import argparse, json
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.utils.paths import RAW, MODELS, REPORTS
from src.features.preprocess import make_Xy, split, build_preprocessor
from src.models.train_supervised import model_zoo, fit_and_eval

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main(args):
    # 1) Load
    df = pd.read_csv(RAW / "customer_churn.csv")

    # 2) X/y, split, preprocessor
    X, y, num_cols, cat_cols = make_Xy(df, drop_noise=True)
    spl = split(X, y, test_size=0.30, val_size=0.50, random_state=42)
    pre = build_preprocessor(num_cols, cat_cols)

    # 3) Train model zoo on TRAIN, select best by VAL ROC-AUC
    zoo = model_zoo(balanced=args.balanced)
    results = fit_and_eval(zoo, pre, spl.X_train, spl.y_train, spl.X_val, spl.y_val)
    results_out = REPORTS / "supervised" / "val_results.csv"
    results_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_out, index=False)
    print("\n=== Validation Results (sorted by ROC-AUC) ===")
    print(results[["model","roc_auc","accuracy","precision_1","recall_1","f1_1"]])

    # 4) Save best validation pipeline (fitted on TRAIN only)
    best_row = results.iloc[0]
    best_name = best_row["model"]
    best_pipe_val = best_row["pipeline"]  # already fitted on TRAIN
    path_val = MODELS / f"{best_name}_VAL.joblib"
    dump(best_pipe_val, path_val)
    print(f"\nSaved validation-selected pipeline (fit on TRAIN): {path_val}")

    # 5) Evaluate best pipeline on VALIDATION & TEST and persist predictions/metrics
    #    (No refit here—use the TRAIN-fitted pipeline for a clean VAL readout)
    y_val_proba = best_pipe_val.predict_proba(spl.X_val)[:,1]
    y_val_pred  = (y_val_proba >= 0.5).astype(int)
    val_metrics = {
        "model": best_name,
        "roc_auc": float(roc_auc_score(spl.y_val, y_val_proba)),
        "report": classification_report(spl.y_val, y_val_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(spl.y_val, y_val_pred).tolist(),
    }
    save_json(val_metrics, REPORTS / "supervised" / f"{best_name}_VAL_metrics.json")
    pd.DataFrame({
        "y_true": spl.y_val.values,
        "y_pred": y_val_pred,
        "y_proba": y_val_proba
    }).to_csv(REPORTS / "supervised" / f"{best_name}_VAL_predictions.csv", index=False)

    # 6) Refit BEST on TRAIN+VAL, then evaluate on TEST (proper final holdout)
    X_train_val = pd.concat([spl.X_train, spl.X_val], axis=0)
    y_train_val = pd.concat([spl.y_train, spl.y_val], axis=0)
    best_cls = model_zoo(balanced=args.balanced)[best_name]
    final_pipe = build_preprocessor(num_cols, cat_cols)
    from sklearn.pipeline import Pipeline as SKLPipeline
    best_full = SKLPipeline([("pre", final_pipe), ("clf", best_cls)])
    best_full.fit(X_train_val, y_train_val)

    # Save final model (fit on TRAIN+VAL)
    path_final = MODELS / f"{best_name}_FINAL.joblib"
    dump(best_full, path_final)
    print(f"Saved final pipeline (fit on TRAIN+VAL): {path_final}")

    # Final TEST evaluation
    y_test_proba = best_full.predict_proba(spl.X_test)[:,1]
    y_test_pred  = (y_test_proba >= 0.5).astype(int)
    test_metrics = {
        "model": best_name,
        "roc_auc": float(roc_auc_score(spl.y_test, y_test_proba)),
        "report": classification_report(spl.y_test, y_test_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(spl.y_test, y_test_pred).tolist(),
    }
    save_json(test_metrics, REPORTS / "supervised" / f"{best_name}_TEST_metrics.json")
    pd.DataFrame({
        "y_true": spl.y_test.values,
        "y_pred": y_test_pred,
        "y_proba": y_test_proba
    }).to_csv(REPORTS / "supervised" / f"{best_name}_TEST_predictions.csv", index=False)

    print("\nArtifacts:")
    print(f"- Validation table: {results_out}")
    print(f"- VAL metrics: {REPORTS / 'supervised' / f'{best_name}_VAL_metrics.json'}")
    print(f"- VAL predictions: {REPORTS / 'supervised' / f'{best_name}_VAL_predictions.csv'}")
    print(f"- FINAL model: {path_final}")
    print(f"- TEST metrics: {REPORTS / 'supervised' / f'{best_name}_TEST_metrics.json'}")
    print(f"- TEST predictions: {REPORTS / 'supervised' / f'{best_name}_TEST_predictions.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn models and save artifacts.")
    parser.add_argument("--balanced", action="store_true", help="Use class_weight='balanced' where supported")
    args = parser.parse_args()
    main(args)
