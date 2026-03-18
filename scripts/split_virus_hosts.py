"""
Split virus accessions into vertebrate-infecting and non-vertebrate-infecting
by querying NCBI for host taxonomy.

Classification strategy (in order):
  1. Use 'host' field from GenBank (most reliable)
  2. Use 'lab_host' field via NCBI taxonomy lookup
  3. Check 'isolation_source' + 'note' for vertebrate keywords
  4. Check virus taxonomy for known vertebrate-associated families
  5. Check virus taxonomy/organism for known non-vertebrate clades
  6. Everything else -> unknown

Usage
-----
    python scripts/split_virus_hosts.py --email your@email.com

Outputs
-------
    data/accessions/virus_vertebrate.txt
    data/accessions/virus_non_vertebrate.txt
    data/accessions/virus_unknown_host.txt   (truly ambiguous)
"""

import argparse
import time
from pathlib import Path

from Bio import Entrez

VERTEBRATE_KEYWORDS = {
    "vertebrata", "mammalia", "aves", "reptilia", "amphibia",
    "actinopterygii", "chondrichthyes", "gnathostomata",
    "tetrapoda", "amniota", "sauropsida", "synapsida",
}

# Virus families that are predominantly vertebrate-infecting.
# Used as a fallback when no host field is available.
VERTEBRATE_VIRUS_TAXONOMY = {
    "herpesvirales", "retroviridae", "orthomyxoviridae", "paramyxoviridae",
    "flaviviridae", "togaviridae", "coronaviridae", "filoviridae",
    "arenaviridae", "bunyavirales", "rhabdoviridae", "picornaviridae",
    "caliciviridae", "reoviridae", "papillomaviridae", "polyomaviridae",
    "adenoviridae", "parvoviridae", "poxviridae", "hepadnaviridae",
}

# Vertebrate keywords in isolation_source / note fields
VERTEBRATE_ISOLATION_HINTS = {
    "human", "patient", "blood", "serum", "plasma", "saliva",
    "nasopharyngeal", "lung", "liver", "brain", "feces",
    "stool", "urine", "swab", "biopsy", "tissue",
    "mouse", "rat", "chicken", "pig", "cow", "horse",
    "dog", "cat", "bat", "monkey", "primate", "fish",
    "sheep", "goat", "deer", "bird", "duck",
}

# Virus taxonomy terms that definitively indicate NON-vertebrate hosts.
# These are well-established clades with known host ranges.
NON_VERTEBRATE_TAXONOMY = {
    # Bacteriophages
    "caudoviricetes",       # tailed bacteriophages (vast majority of phages)
    "microviridae",         # small ssDNA bacteriophages
    "inoviridae",           # filamentous bacteriophages
    "tectiviridae",         # bacteriophages
    "corticoviridae",       # bacteriophages
    "plasmaviridae",        # mycoplasma phages
    "leviviricetes",        # RNA bacteriophages (Leviviridae etc.)
    "cystoviridae",         # dsRNA bacteriophages
    # Plant viruses
    "geminiviridae",        # plant viruses (begomoviruses, mastreviruses, etc.)
    "nanoviridae",          # plant viruses
    "caulimoviridae",       # plant pararetroviruses
    "virgaviridae",         # plant viruses (TMV-like)
    "bromoviridae",         # plant viruses
    "closteroviridae",      # plant viruses
    "potyviridae",          # plant viruses
    "tombusviridae",        # plant viruses
    "luteoviridae",         # plant viruses
    "partitiviridae",       # plant/fungal viruses
    "tymoviridae",          # plant viruses
    "secoviridae",          # plant viruses
    "solemoviridae",        # plant viruses
    "kitaviridae",          # plant viruses
    "fimoviridae",          # plant viruses
    "tospoviridae",         # plant viruses (thrips-transmitted)
    "betaflexiviridae",     # plant viruses
    "alphaflexiviridae",    # plant viruses
    # Fungal viruses
    "chrysoviridae",        # fungal viruses
    "totiviridae",          # fungal/protozoan viruses
    "hypoviridae",          # fungal viruses
    "megabirnaviridae",     # fungal viruses
    "quadriviridae",        # fungal viruses
    # Archaeal viruses
    "ligamenvirales",       # archaeal viruses
    "rudiviridae",          # archaeal viruses
    "lipothrixviridae",     # archaeal viruses
    "fuselloviridae",       # archaeal viruses
    "bicaudaviridae",       # archaeal viruses
    "ampullaviridae",       # archaeal viruses
    "globuloviridae",       # archaeal viruses
    "guttaviridae",         # archaeal viruses
    "turriviridae",         # archaeal viruses
    # Insect-specific viruses
    "baculoviridae",        # insect viruses
    "nudiviridae",          # insect viruses
    "polydnaviridae",       # insect viruses (parasitoid wasps)
    "polydnaviriformidae",  # parasitoid wasp viruses
    "bracoviriform",        # braconid wasp viruses
    "iflaviridae",          # insect viruses
    "dicistroviridae",      # insect viruses
    "bidnaviridae",         # insect viruses (Bombyx mori densovirus)
    "genomoviridae",        # fungal/environmental ssDNA viruses
    # Diatom/algal viruses
    "bacilladnaviridae",    # diatom viruses
    "phycodnaviridae",      # algal viruses
    # Arthropod-associated ssDNA viruses
    "smacoviridae",         # arthropod-associated circular viruses
    # Giant viruses (amoeba/protist hosts)
    "nucleocytoviricota",   # giant viruses (Mimivirus, Marseillevirus, etc.)
    # Additional archaeal viruses
    "pleolipoviridae",      # archaeal viruses
    "haloruvirales",        # archaeal viruses
    # Plant virus satellites
    "tolecusatellitidae",   # begomovirus satellites (plant)
}

# Organism name keywords that indicate non-vertebrate viruses
NON_VERTEBRATE_ORGANISM = {
    "phage", "bacteriophage",
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


def is_non_vertebrate_by_taxonomy(taxonomy: str, organism: str) -> bool:
    """Check if the virus taxonomy or organism name definitively indicates
    a non-vertebrate host (bacteriophage, plant virus, archaeal virus, etc.)."""
    tax_lower = taxonomy.lower()
    org_lower = organism.lower()

    # Check organism name for phage keywords
    for kw in NON_VERTEBRATE_ORGANISM:
        if kw in org_lower:
            return True

    # Check taxonomy for known non-vertebrate viral clades
    for clade in NON_VERTEBRATE_TAXONOMY:
        if clade in tax_lower:
            return True

    return False


def fetch_host_info_batch(accessions: list[str], batch_size: int = 200) -> dict:
    """Fetch host info for a batch of accessions from GenBank."""
    results = {}
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        ids = ",".join(batch)
        records = None
        for attempt in range(1, 4):
            try:
                handle = Entrez.efetch(
                    db="nucleotide", id=ids, rettype="gb", retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                break
            except Exception as e:
                if attempt < 3:
                    wait = 10 * attempt
                    print(f"  Retry {attempt}/3 for batch {i}-{i+len(batch)} "
                          f"(error: {e}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Failed after 3 retries for batch {i}-{i+len(batch)}: {e}")
        if records is None:
            continue

        for rec in records:
            acc = rec.get("GBSeq_primary-accession", "")
            version = rec.get("GBSeq_accession-version", acc)
            host = ""
            lab_host = ""
            isolation_source = ""
            note = ""
            organism = rec.get("GBSeq_organism", "")
            taxonomy = rec.get("GBSeq_taxonomy", "")
            for qual in rec.get("GBSeq_feature-table", [{}])[0].get("GBFeature_quals", []):
                name = qual.get("GBQualifier_name", "")
                value = qual.get("GBQualifier_value", "")
                if name == "host":
                    host = value
                elif name == "lab_host":
                    lab_host = value
                elif name == "isolation_source":
                    isolation_source = value
                elif name == "note":
                    note = value
            results[version] = {
                "host": host,
                "lab_host": lab_host,
                "isolation_source": isolation_source,
                "note": note,
                "organism": organism,
                "taxonomy": taxonomy,
            }

        print(f"  Fetched {min(i + batch_size, len(accessions))}/{len(accessions)}")
        time.sleep(0.4)

    return results


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

    print("\nClassifying accessions by host...")
    for acc in accessions:
        info = host_info.get(acc, {})
        host = info.get("host", "")
        organism = info.get("organism", "")
        taxonomy = info.get("taxonomy", "")

        # Strategy 1: Use host field (most reliable)
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
            continue

        # Strategy 2: Use lab_host field
        lab_host = info.get("lab_host", "")
        if lab_host:
            if lab_host not in host_cache:
                tax_id = get_host_taxid(lab_host)
                if tax_id:
                    lineage = get_host_lineage(tax_id)
                    host_cache[lab_host] = is_vertebrate_host(lineage or "")
                    time.sleep(0.35)
                else:
                    lh_lower = lab_host.lower()
                    host_cache[lab_host] = any(kw in lh_lower for kw in {
                        "human", "homo sapiens", "mouse", "mus musculus",
                        "chicken", "gallus", "pig", "sus scrofa", "cow",
                        "bos taurus", "dog", "cat", "horse", "sheep",
                        "goat", "rat", "monkey", "primate", "bat",
                        "fish", "salmon", "trout", "vero", "hela",
                        "hep", "mdck", "bhk", "cho",
                    })

            if host_cache[lab_host]:
                vertebrate.append(acc)
            else:
                non_vertebrate.append(acc)
            continue

        # Strategy 3: Check isolation_source + note for vertebrate keywords
        isolation_source = info.get("isolation_source", "")
        note = info.get("note", "")
        combined_text = f"{isolation_source} {note}".lower()
        if any(kw in combined_text for kw in VERTEBRATE_ISOLATION_HINTS):
            vertebrate.append(acc)
            continue

        # Strategy 4: Check virus taxonomy for known vertebrate families
        if any(fam in taxonomy.lower() for fam in VERTEBRATE_VIRUS_TAXONOMY):
            vertebrate.append(acc)
            continue

        # Strategy 5: Check virus taxonomy/organism for non-vertebrate clades
        if is_non_vertebrate_by_taxonomy(taxonomy, organism):
            non_vertebrate.append(acc)
            continue

        # Strategy 6: Truly ambiguous — no host, no clear taxonomy signal
        unknown.append(acc)

    outdir = Path(args.outdir)
    (outdir / "virus_vertebrate.txt").write_text("\n".join(vertebrate) + "\n")
    (outdir / "virus_non_vertebrate.txt").write_text("\n".join(non_vertebrate) + "\n")
    (outdir / "virus_unknown_host.txt").write_text("\n".join(unknown) + "\n")

    print(f"\nResults:")
    print(f"  Vertebrate-infecting:     {len(vertebrate)}")
    print(f"  Non-vertebrate-infecting: {len(non_vertebrate)}")
    print(f"  Unknown host:             {len(unknown)}")
    print(f"\nSaved to {outdir}/")


if __name__ == "__main__":
    main()
