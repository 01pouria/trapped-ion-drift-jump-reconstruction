#!/usr/bin/env python
from pathlib import Path

from trapped_ion_pdmp.validation import simulator_validation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "generated"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = simulator_validation()
    table.to_csv(OUT / "simulator_validation.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
