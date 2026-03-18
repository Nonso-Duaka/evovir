"""
Build metadata.csv from downloaded FASTA subdirectories.

Three modes:
  --mode binary_vv       : vertebrate virus (1) vs non-vertebrate virus (0)
  --mode binary_vv_all   : vertebrate virus (1) vs everything else (0)
  --mode multiclass      : vertebrate virus, non-vertebrate virus, other (3 classes)

Usage
-----
    python scripts/build_metadata.py --mode binary_vv
    python scripts/build_metadata.py --mode binary_vv_all
    python scripts/build_metadata.py --mode multiclass
"""

import argparse
from pathlib import Path

import pandas as pd


ALL_SUBDIRS = [
    "virus_vertebrate",
    "virus_non_vertebrate",
    "bacteria",
    "archaea",
    "protozoa",
    "fungi",
    "plasmid",
]

NON_VIRUS = ["bacteria", "archaea", "protozoa", "fungi", "plasmid"]


def get_label(subdir_name: str, mode: str):
    """Return the label for a subdirectory given the mode."""
    if mode == "binary_vv":
        if subdir_name == "virus_vertebrate":
            return 1
        elif subdir_name == "virus_non_vertebrate":
            return 0
        else:
            return None  # skip non-virus groups

    elif mode == "binary_vv_all":
        if subdir_name == "virus_vertebrate":
            return 1
        else:
            return 0  # everything else is negative

    elif mode == "multiclass":
        if subdir_name == "virus_vertebrate":
            return "virus_vertebrate"
        elif subdir_name == "virus_non_vertebrate":
            return "virus_non_vertebrate"
        else:
            return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta-dir", default="data/fasta", help="Root FASTA directory")
    parser.add_argument("--output", default=None, help="Output CSV path (auto-named if omitted)")
    parser.add_argument("--mode", required=True,
                        choices=["binary_vv", "binary_vv_all", "multiclass"],
                        help="binary_vv: vv vs non-vv only | "
                             "binary_vv_all: vv vs all others | "
                             "multiclass: vv, non-vv, other")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/metadata_{args.mode}.csv"

    fasta_root = Path(args.fasta_dir)
    rows = []

    for subdir_name in ALL_SUBDIRS:
        label = get_label(subdir_name, args.mode)
        if label is None:
            continue

        subdir = fasta_root / subdir_name
        if not subdir.exists():
            print(f"Skipping {subdir} (not found)")
            continue

        fa_files = list(subdir.glob("*.fa"))
        print(f"{subdir_name} -> {label}: {len(fa_files)} sequences")

        for fa in fa_files:
            accession = fa.stem
            rows.append({
                "accession": accession,
                "label": label,
                "fasta_file": str(fa),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"\nTotal sequences: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
