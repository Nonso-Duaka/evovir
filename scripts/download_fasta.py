"""
Download FASTA sequences from NCBI given an accession list.

Usage
-----
    python scripts/download_fasta.py --email your@email.com \
        --accessions data/accessions/virus_vertebrate.txt \
        --outdir data/fasta/vertebrate

    # Or download all splits at once:
    python scripts/download_fasta.py --email your@email.com --all
"""

import argparse
import time
from pathlib import Path

from Bio import Entrez


def download_batch(accessions: list[str], outdir: Path, batch_size: int = 100) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    total = len(accessions)

    for i in range(0, total, batch_size):
        batch = accessions[i:i + batch_size]
        ids = ",".join(batch)
        try:
            handle = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
            fasta_text = handle.read()
            handle.close()
        except Exception as e:
            print(f"  Error at batch {i}-{i+len(batch)}: {e}")
            time.sleep(10)
            continue

        # Write each record to its own file
        current_acc = None
        current_lines = []
        for line in fasta_text.strip().split("\n"):
            if line.startswith(">"):
                if current_acc and current_lines:
                    (outdir / f"{current_acc}.fa").write_text("\n".join(current_lines) + "\n")
                current_acc = line.split()[0].lstrip(">").split(".")[0]
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_acc and current_lines:
            (outdir / f"{current_acc}.fa").write_text("\n".join(current_lines) + "\n")

        print(f"  Downloaded {min(i + batch_size, total)}/{total}")
        time.sleep(0.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="NCBI Entrez email")
    parser.add_argument("--accessions", type=str, help="Path to accession list file")
    parser.add_argument("--outdir", type=str, help="Output directory for FASTA files")
    parser.add_argument("--all", action="store_true", help="Download all splits (vertebrate, non-vertebrate, negatives)")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    Entrez.email = args.email

    if args.all:
        acc_dir = Path("data/accessions")
        jobs = [
            (acc_dir / "virus_vertebrate.txt", Path("data/fasta")),
            (acc_dir / "virus_non_vertebrate.txt", Path("data/fasta")),
            (acc_dir / "bacteria.txt", Path("data/fasta")),
        ]
        for acc_file, out in jobs:
            if not acc_file.exists():
                print(f"Skipping {acc_file} (not found)")
                continue
            accs = acc_file.read_text().strip().splitlines()
            print(f"\nDownloading {len(accs)} sequences from {acc_file.name}...")
            download_batch(accs, out, args.batch_size)
    else:
        if not args.accessions or not args.outdir:
            print("Provide --accessions and --outdir, or use --all")
            return
        accs = Path(args.accessions).read_text().strip().splitlines()
        print(f"Downloading {len(accs)} sequences...")
        download_batch(accs, Path(args.outdir), args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
