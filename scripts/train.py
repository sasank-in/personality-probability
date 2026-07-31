"""
Training script — Big Five personality trait classifier.

Fits a regularized linear model (== 5 L2-penalized logistic regressions) on
Mistral text embeddings from the essays-big5 dataset, folds the coefficients
into PersonalityClassifier, and saves the checkpoint + scaler into artifacts/.

With only ~1.6k training essays, deeper MLPs overfit (train AUC ~1.0, test
AUC ~0.6) and score worse on held-out data than this linear model
(~0.62 mean acc vs ~0.58).

Run:  python scripts/train.py     (needs: pip install ".[train,mistral]")
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Make the package importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from personality_api.model import PersonalityClassifier  # noqa: E402
from personality_api.traits import TRAIT_KEYS, TRAIT_NAMES  # noqa: E402

np.random.seed(42)

ARTIFACTS = ROOT / "artifacts"
C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.03]  # inverse L2 strength; smaller = more regularized
DATASET = "jingjietan/essays-big5-mistral-embeddings"


def convert_split(split):
    X = np.array(split["embedding"], dtype=np.float32)
    y = np.column_stack([np.array(split[t], dtype=np.float32) for t in TRAIT_KEYS])
    return X, y


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    print("Loading dataset...")
    ds = load_dataset(DATASET)

    X_train, y_train = convert_split(ds["train"])
    X_val, y_val = convert_split(ds["validation"])
    X_test, y_test = convert_split(ds["evaluation"])
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.vstack([y_train, y_val])
    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    W = np.zeros((5, X_fit_s.shape[1]), dtype=np.float32)
    b = np.zeros(5, dtype=np.float32)
    chosen_C = []
    print("\nFitting per-trait logistic regression (C tuned on validation):")
    for i, name in enumerate(TRAIT_NAMES):
        best_acc, best_C = -1.0, None
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=3000).fit(X_train_s, y_train[:, i])
            vacc = accuracy_score(y_val[:, i], clf.predict(X_val_s))
            if vacc > best_acc:
                best_acc, best_C = vacc, C
        clf = LogisticRegression(C=best_C, max_iter=3000).fit(X_fit_s, y_fit[:, i])
        W[i] = clf.coef_[0]
        b[i] = clf.intercept_[0]
        chosen_C.append(best_C)
        print(f"  {name:20s} C={best_C:<7} val_acc={best_acc:.4f}")

    model = PersonalityClassifier(input_dim=X_fit_s.shape[1])
    with torch.no_grad():
        model.fc_out.weight.copy_(torch.tensor(W))
        model.fc_out.bias.copy_(torch.tensor(b))
    model.eval()

    with torch.no_grad():
        test_probs = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32))).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    trait_accs = [accuracy_score(y_test[:, i], test_preds[:, i]) for i in range(5)]
    test_acc = accuracy_score(y_test.flatten(), test_preds.flatten())
    exact_match = np.all(test_preds == y_test, axis=1).mean()

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Exact Match      : {exact_match:.4f} ({exact_match * 100:.2f}%)\n")
    for name, acc in zip(TRAIT_NAMES, trait_accs, strict=True):
        print(f"  {name:20s}: {acc:.4f} ({acc * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("PER-TRAIT CLASSIFICATION REPORTS")
    print("=" * 60)
    for i, name in enumerate(TRAIT_NAMES):
        print(f"\n{name}:")
        print(
            classification_report(
                y_test[:, i], test_preds[:, i], target_names=["Low", "High"], zero_division=0
            )
        )

    joblib.dump(scaler, ARTIFACTS / "x_scaler.pkl")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_fit_s.shape[1],
            "test_accuracy": test_acc,
            "trait_names": list(TRAIT_NAMES),
            "trait_accuracies": trait_accs,
            "model_type": "logistic_regression_linear",
            "chosen_C": chosen_C,
        },
        ARTIFACTS / "personality_model.pt",
    )
    print(f"\nSaved: {ARTIFACTS / 'personality_model.pt'}, {ARTIFACTS / 'x_scaler.pkl'}")


if __name__ == "__main__":
    main()
