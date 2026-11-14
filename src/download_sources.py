"""Download the public source files of the benchmark into data/raw/.

Each file is fetched from its official location at a fixed revision and checked against its SHA-256;
a file that is already present with the right checksum is not fetched again.

    python src/download_sources.py            # fetch and verify
    python src/download_sources.py --check    # only verify files already in data/raw/

Sources (see README.md for the licences):
  AmbiK           github.com/cog-model/AmbiK-dataset, commit 9d4f60d4 (no licence stated; not redistributed)
  SafeAgentBench  huggingface.co/datasets/safeagentbench/SafeAgentBench, revision f5d2df64 (CC BY 4.0)
  EmbodyGuard     github.com/Yonsei-MIR/EAI-safety, commit 14231716 (MIT)
  TRUST-NAV       github.com/iitis/Risk-Aware-Semantic-Grounding, commit 82e88056 (MIT)
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GH = "https://raw.githubusercontent.com"
AMBIK = f"{GH}/cog-model/AmbiK-dataset/9d4f60d4224b4183cd35d11233d8114aeaefc2f6"
SAB = "https://huggingface.co/datasets/safeagentbench/SafeAgentBench/resolve/f5d2df6473e519a4f9ece5a17fe8b755d8256819"
EG = f"{GH}/Yonsei-MIR/EAI-safety/14231716b610a96871bf3f2fcb546f293cac8f33"
TN = f"{GH}/iitis/Risk-Aware-Semantic-Grounding/82e88056f05eb130de21c5371a68fea543e59b16"

# (local path under data/raw, URL, SHA-256, size in bytes)
FILES = [
    ("ambik/AmbiK_data.csv", f"{AMBIK}/AmbiK_data.csv",
     "98bb4677eb7fad00a58393aa751f833c5e01dcda9090fe0ef202272b20e2f531", 1848398),
    ("safeagentbench/safe_detailed.jsonl", f"{SAB}/safe_detailed.jsonl",
     "6d86acf164ff1caece3dcc145ab34ad248896e5d1fe2fe353b0fad18c0560e40", 93037),
    ("embodyguard/malicious_instructions_by_task.json",
     f"{EG}/src/behavior_eval/evaluation/refusal/resources/data/instructions_by_task.json",
     "7d4354378552e920faa2aa451c3be502e6eb819fd315f9761434664f1fcd21ce", 1154087),
    ("trustnav/dataset_query_decision_gt.json", f"{TN}/dataset/dataset_query_decision_gt.json",
     "8e57040370174c950f7bd1a377bb5e864326bc66ef5f949cc82b29b3670b9c03", 49868),
    ("trustnav/smart_home_map.json", f"{TN}/dataset/smart_home_map.json",
     "331590d882f43703b4f1aed97ca963bbd78c11c139b1a15996fc112500b51f2c", 14516),
    # the TRUST-NAV decision module whose rules analyze.py reimplements (TRUSTNAV_GATES); not read by the code
    ("trustnav/decision_layer.py", f"{TN}/decision_layer.py",
     "ede9b0b11f8863e431c91f59321045d3c433a7fbc5da1bfa11bc8ceb7ab5538c", 5076),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the files in data/raw/ without downloading")
    args = ap.parse_args(argv)
    bad = 0
    for rel, url, digest, size in FILES:
        path = RAW / rel
        if path.exists() and sha256(path) == digest:
            print(f"ok          {rel}")
            continue
        if args.check:
            print(f"{'MISMATCH' if path.exists() else 'MISSING':11} {rel}")
            bad += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel} ({size:,} bytes) from {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != digest:
            print(f"CHECKSUM MISMATCH for {rel}: not saved")
            bad += 1
            continue
        path.write_bytes(data)
    if bad:
        sys.exit(f"{bad} file(s) missing or with a wrong checksum")
    print("all source files present and verified")


if __name__ == "__main__":
    main()
