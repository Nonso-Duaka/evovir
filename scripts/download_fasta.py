"""
Download FASTA sequences from NCBI given an accession list.

Handles two accession types:
  - Nucleotide IDs (NC_*, NW_*, NZ_*, etc.) -> fetched directly from nucleotide db
  - Assembly IDs (GCF_*, GCA_*) -> resolved to nucleotide IDs via NCBI Assembly db

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


MAX_RETRIES = 3


def fetch_with_retry(fetch_fn, description: str):
    """Call fetch_fn() with retries on failure. Returns result or raises on final failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_fn()
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} for {description} "
                      f"(error: {e}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed after {MAX_RETRIES} retries for {description}: {e}")
                raise


def is_assembly_accession(acc: str) -> bool:
    """Check if an accession is a GCF/GCA assembly ID."""
    return acc.startswith("GCF_") or acc.startswith("GCA_")


def resolve_assembly_to_nucleotide(assembly_acc: str) -> list[str]:
    """Resolve a GCF/GCA assembly accession to its nucleotide sequence IDs."""
    try:
        def _search():
            h = Entrez.esearch(db="assembly", term=assembly_acc)
            r = Entrez.read(h)
            h.close()
            return r

        result = fetch_with_retry(_search, f"assembly search {assembly_acc}")

        if not result["IdList"]:
            return []

        assembly_id = result["IdList"][0]

        def _link():
            h = Entrez.elink(dbfrom="assembly", db="nucleotide", id=assembly_id)
            r = Entrez.read(h)
            h.close()
            return r

        links = fetch_with_retry(_link, f"assembly link {assembly_acc}")

        nuc_ids = []
        for linkset in links:
            for link_db in linkset.get("LinkSetDb", []):
                for link in link_db.get("Link", []):
                    nuc_ids.append(link["Id"])

        return nuc_ids
    except Exception as e:
        print(f"  Error resolving assembly {assembly_acc}: {e}")
        return []


def download_nucleotide_batch(accessions: list[str], outdir: Path, batch_size: int = 100) -> None:
    """Download FASTA for nucleotide accessions (NC_, NW_, NZ_, etc.)."""
    total = len(accessions)

    for i in range(0, total, batch_size):
        batch = accessions[i:i + batch_size]
        ids = ",".join(batch)

        def _fetch(ids=ids):
            h = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
            text = h.read()
            h.close()
            return text

        try:
            fasta_text = fetch_with_retry(_fetch, f"nucleotide batch {i}-{i+len(batch)}")
        except Exception:
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


def download_assembly_batch(accessions: list[str], outdir: Path, batch_size: int = 100) -> None:
    """Download FASTA for assembly accessions (GCF_, GCA_) by resolving to nucleotide IDs."""
    total = len(accessions)

    for idx, assembly_acc in enumerate(accessions):
        print(f"  Resolving assembly {idx + 1}/{total}: {assembly_acc}...")
        nuc_ids = resolve_assembly_to_nucleotide(assembly_acc)
        time.sleep(0.35)

        if not nuc_ids:
            print(f"    No nucleotide sequences found for {assembly_acc}, skipping")
            continue

        # Download the nucleotide sequences for this assembly
        # Use the assembly accession as the filename (concatenate all contigs)
        try:
            # Fetch in sub-batches if many contigs
            all_lines = []
            for i in range(0, len(nuc_ids), batch_size):
                sub_batch = nuc_ids[i:i + batch_size]
                ids = ",".join(sub_batch)

                def _fetch(ids=ids):
                    h = Entrez.efetch(db="nucleotide", id=ids, rettype="fasta", retmode="text")
                    text = h.read()
                    h.close()
                    return text

                fasta_text = fetch_with_retry(_fetch, f"contigs for {assembly_acc} batch {i}")
                all_lines.append(fasta_text.strip())
                time.sleep(0.4)

            # Write as a single multi-record FASTA per assembly
            clean_acc = assembly_acc.split(".")[0]
            (outdir / f"{clean_acc}.fa").write_text("\n".join(all_lines) + "\n")
            print(f"    Saved {len(nuc_ids)} contigs for {assembly_acc}")

        except Exception as e:
            print(f"    Error downloading sequences for {assembly_acc}: {e}")
            time.sleep(5)


def download_batch(accessions: list[str], outdir: Path, batch_size: int = 100) -> None:
    """Route accessions to the correct download method."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Split into nucleotide vs assembly accessions
    nuc_accs = [a for a in accessions if not is_assembly_accession(a)]
    asm_accs = [a for a in accessions if is_assembly_accession(a)]

    if nuc_accs:
        print(f"  Downloading {len(nuc_accs)} nucleotide accessions...")
        download_nucleotide_batch(nuc_accs, outdir, batch_size)

    if asm_accs:
        print(f"  Resolving and downloading {len(asm_accs)} assembly accessions...")
        download_assembly_batch(asm_accs, outdir, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="NCBI Entrez email")
    parser.add_argument("--accessions", type=str, help="Path to accession list file")
    parser.add_argument("--outdir", type=str, help="Output directory for FASTA files")
    parser.add_argument("--all", action="store_true", help="Download all splits")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    Entrez.email = args.email

    if args.all:
        acc_dir = Path("data/accessions")
        jobs = [
            (acc_dir / "virus_vertebrate.txt", Path("data/fasta/virus_vertebrate")),
            (acc_dir / "virus_non_vertebrate.txt", Path("data/fasta/virus_non_vertebrate")),
            (acc_dir / "bacteria.txt", Path("data/fasta/bacteria")),
            (acc_dir / "archaea.txt", Path("data/fasta/archaea")),
            (acc_dir / "protozoa.txt", Path("data/fasta/protozoa")),
            (acc_dir / "fungi.txt", Path("data/fasta/fungi")),
            (acc_dir / "plasmid.txt", Path("data/fasta/plasmid")),
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
