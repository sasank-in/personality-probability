"""
Download the essays-big5 dataset (raw text + Big Five labels) into data/.

Saves one CSV per split so the corpus is versioned locally and training does
not depend on a live Hugging Face connection.

Run:  python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET = "jingjietan/essays-big5"  # raw `text` + O,C,E,A,N labels
COLUMNS = ["id", "text", "O", "C", "E", "A", "N", "ptype"]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    print(f"Downloading {DATASET} ...")
    ds = load_dataset(DATASET)

    # HF split name -> our file name
    split_map = {"train": "train", "validation": "validation", "test": "test"}
    for hf_split, out_name in split_map.items():
        split = ds[hf_split]
        # Stable id: prefer the original index column if present.
        id_col = "__index_level_0__" if "__index_level_0__" in split.column_names else None
        df = split.to_pandas()
        if id_col:
            df = df.rename(columns={id_col: "id"})
        elif "id" not in df.columns:
            df.insert(0, "id", range(len(df)))
        keep = [c for c in COLUMNS if c in df.columns]
        df = df[keep]
        out = DATA / f"{out_name}.csv"
        df.to_csv(out, index=False)
        print(f"  {out_name:11s} {len(df):5d} rows -> {out}")

    print(f"\nDone. Dataset saved under {DATA}")


if __name__ == "__main__":
    main()
