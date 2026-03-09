"""
Download viral genome sequences from NCBI and build a labelled dataset.

Usage
-----
    python scripts/download_data.py --config configs/default.yaml

What it does
------------
1. For each class (vertebrate / non-vertebrate) it searches NCBI Nucleotide
   for virus sequences with matching host taxonomy.
2. Downloads sequences in batches via Entrez efetch.
3. Writes:
      data/fasta/vertebrate/*.fa
      data/fasta/non_vertebrate/*.fa
      data/metadata.csv

Requirements
------------
- Set `ncbi_email` in your config (NCBI requires a valid email for Entrez).
- Biopython must be installed.
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from Bio import Entrez, SeqIO


def _build_query(host_taxids: List[int]) -> str:
    host_parts = " OR ".join(f"txid{tid}[Host]" for tid in host_taxids)
    return (
        f'("Viruses"[Organism]) AND ({host_parts}) '
        f'AND ("complete genome"[Title] OR "complete sequence"[Title])'
    )


def _search_ids(query: str, max_results: int) -> List[str]:
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_results, idtype="acc")
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def _fetch_sequences(ids: List[str], out_dir: Path, label: int, label_name: str) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    batch_size = 200

    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        print(f"  Fetching {i+1}–{i+len(batch)} of {len(ids)}…", end=" ", flush=True)
        try:
            handle = Entrez.efetch(
                db="nucleotide", id=",".join(batch), rettype="fasta", retmode="text"
            )
            records = list(SeqIO.parse(handle, "fasta"))
            handle.close()
        except Exception as exc:
            print(f"WARN: fetch failed ({exc}), skipping batch.")
            time.sleep(5)
            continue

        for rec in records:
            accession = rec.id.split(".")[0]
            fa_path = out_dir / f"{accession}.fa"
            with open(fa_path, "w") as fh:
                SeqIO.write(rec, fh, "fasta")
            metadata.append(
                {
                    "accession": accession,
                    "label": label,
                    "label_name": label_name,
                    "length": len(rec.seq),
                    "fasta_file": f"{label_name}/{accession}.fa",
                    "description": rec.description,
                }
            )
        print(f"done ({len(records)} records).")
        time.sleep(0.34)  # NCBI rate limit: max 3 req/s without API key

    return metadata


def main(config_path: str, dry_run: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    Entrez.email = cfg["ncbi_email"]
    if Entrez.email == "your@email.com":
        raise ValueError("Set 'ncbi_email' in your config before downloading.")

    data_dir = Path(cfg["data_dir"])
    max_per_class: int = cfg.get("max_per_class") or 10_000
    all_metadata = []

    print("\n[1/2] Searching vertebrate-infecting viruses…")
    ids_vert = _search_ids(_build_query(cfg["vertebrate_host_taxids"]), max_per_class)
    print(f"  Found {len(ids_vert)} records.")
    if not dry_run:
        all_metadata.extend(
            _fetch_sequences(ids_vert, data_dir / "fasta" / "vertebrate", label=1, label_name="vertebrate")
        )

    print("\n[2/2] Searching non-vertebrate-infecting viruses…")
    ids_nonvert = _search_ids(_build_query(cfg["non_vertebrate_host_taxids"]), max_per_class)
    print(f"  Found {len(ids_nonvert)} records.")
    if not dry_run:
        all_metadata.extend(
            _fetch_sequences(ids_nonvert, data_dir / "fasta" / "non_vertebrate", label=0, label_name="non_vertebrate")
        )

    if not dry_run and all_metadata:
        meta_path = data_dir / "metadata.csv"
        pd.DataFrame(all_metadata).to_csv(meta_path, index=False)
        n_vert = sum(r["label"] == 1 for r in all_metadata)
        n_nonvert = sum(r["label"] == 0 for r in all_metadata)
        print(f"\nSaved metadata → {meta_path}")
        print(f"  Vertebrate:     {n_vert}")
        print(f"  Non-vertebrate: {n_nonvert}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.config, dry_run=args.dry_run)
