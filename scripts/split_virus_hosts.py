"""
Split virus accessions into vertebrate-infecting and non-vertebrate-infecting
by querying NCBI for host taxonomy.

Usage
-----
    python scripts/split_virus_hosts.py --email your@email.com

Outputs
-------
    data/accessions/virus_vertebrate.txt
    data/accessions/virus_non_vertebrate.txt
    data/accessions/virus_unknown_host.txt   (no host annotation found)
"""

import argparse
import time
from pathlib import Path

from Bio import Entrez, SeqIO

VERTEBRATE_CLASSES = {
    "Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopteri",
    "Chondrichthyes", "Cladistia", "Coelacanthi", "Dipnoi",
    "Hyperotreti", "Petromyzonti", "Teleostei", "Sarcopterygii",
}

VERTEBRATE_KEYWORDS = {
    "vertebrata", "mammalia", "aves", "reptilia", "amphibia",
    "actinopterygii", "chondrichthyes", "gnathostomata",
    "tetrapoda", "amniota", "sauropsida", "synapsida",
}


def is_vertebrate_host(host_lineage: str) -> bool:
    lineage_lower = host_lineage.lower()
    return any(kw in lineage_lower for kw in VERTEBRATE_KEYWORDS)


def get_host_lineage(tax_id: str) -> str | None:
    try:
        handle = Entrez.efetch(db="taxonomy", id=tax_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        if records:
            return records[0].get("Lineage", "")
    except Exception:
        pass
    return None


def get_host_taxid(host_name: str) -> str | None:
    try:
        handle = Entrez.esearch(db="taxonomy", term=host_name)
        result = Entrez.read(handle)
        handle.close()
        if result["IdList"]:
            return result["IdList"][0]
    except Exception:
        pass
    return None


def fetch_host_info_batch(accessions: list[str], batch_size: int = 200) -> dict:
    """Fetch host info for a batch of accessions from GenBank."""
    results = {}
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        ids = ",".join(batch)
        try:
            handle = Entrez.efetch(
                db="nucleotide", id=ids, rettype="gb", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"  Error fetching batch {i}-{i+len(batch)}: {e}")
            time.sleep(5)
            continue

        for rec in records:
            acc = rec.get("GBSeq_primary-accession", "")
            version = rec.get("GBSeq_accession-version", acc)
            host = ""
            taxonomy = rec.get("GBSeq_taxonomy", "")
            for qual in rec.get("GBSeq_feature-table", [{}])[0].get("GBFeature_quals", []):
                if qual.get("GBQualifier_name") == "host":
                    host = qual.get("GBQualifier_value", "")
                    break
            results[version] = {"host": host, "taxonomy": taxonomy}

        print(f"  Fetched {min(i + batch_size, len(accessions))}/{len(accessions)}")
        time.sleep(0.4)

    return results


def classify_by_taxonomy(taxonomy: str) -> str | None:
    """Fallback: check if the virus itself is in a vertebrate-associated taxonomy."""
    tax_lower = taxonomy.lower()
    vertebrate_virus_families = {
        "herpesvirales", "retroviridae", "orthomyxoviridae", "paramyxoviridae",
        "flaviviridae", "togaviridae", "coronaviridae", "filoviridae",
        "arenaviridae", "bunyavirales", "rhabdoviridae", "picornaviridae",
        "caliciviridae", "reoviridae", "papillomaviridae", "polyomaviridae",
        "adenoviridae", "parvoviridae", "poxviridae", "hepadnaviridae",
    }
    for fam in vertebrate_virus_families:
        if fam in tax_lower:
            return "vertebrate"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="NCBI Entrez email")
    parser.add_argument("--input", default="data/accessions/virus.txt")
    parser.add_argument("--outdir", default="data/accessions")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    Entrez.email = args.email
    Entrez.api_key = None  # set if you have one for faster queries

    accessions = Path(args.input).read_text().strip().splitlines()
    print(f"Total virus accessions: {len(accessions)}")

    print("\nFetching host info from NCBI (this will take a while)...")
    host_info = fetch_host_info_batch(accessions, batch_size=args.batch_size)

    vertebrate = []
    non_vertebrate = []
    unknown = []
    host_cache = {}

    for acc in accessions:
        info = host_info.get(acc, {})
        host = info.get("host", "")
        taxonomy = info.get("taxonomy", "")

        if host:
            if host not in host_cache:
                tax_id = get_host_taxid(host)
                if tax_id:
                    lineage = get_host_lineage(tax_id)
                    host_cache[host] = is_vertebrate_host(lineage or "")
                    time.sleep(0.35)
                else:
                    host_lower = host.lower()
                    host_cache[host] = any(kw in host_lower for kw in {
                        "human", "homo sapiens", "mouse", "mus musculus",
                        "chicken", "gallus", "pig", "sus scrofa", "cow",
                        "bos taurus", "dog", "cat", "horse", "sheep",
                        "goat", "rat", "monkey", "primate", "bat",
                        "fish", "salmon", "trout",
                    })

            if host_cache[host]:
                vertebrate.append(acc)
            else:
                non_vertebrate.append(acc)
        else:
            result = classify_by_taxonomy(taxonomy)
            if result == "vertebrate":
                vertebrate.append(acc)
            else:
                unknown.append(acc)

    outdir = Path(args.outdir)
    (outdir / "virus_vertebrate.txt").write_text("\n".join(vertebrate) + "\n")
    (outdir / "virus_non_vertebrate.txt").write_text("\n".join(non_vertebrate) + "\n")
    (outdir / "virus_unknown_host.txt").write_text("\n".join(unknown) + "\n")

    print(f"\nResults:")
    print(f"  Vertebrate-infecting:     {len(vertebrate)}")
    print(f"  Non-vertebrate-infecting: {len(non_vertebrate)}")
    print(f"  Unknown host:             {len(unknown)}")
    print(f"\nSaved to {outdir}/virus_vertebrate.txt and virus_non_vertebrate.txt")


if __name__ == "__main__":
    main()
