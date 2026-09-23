"""Thin CLI wrapper around src.neo4j_export. Run from the repo root, e.g.:

    python -m scripts.run_neo4j_export --in data/conflicts/train-000000.json --dry-run
    NEO4J_PASSWORD=secret python -m scripts.run_neo4j_export \
        --in data/conflicts/train-000000.json --wipe
"""
from src.neo4j_export import main

if __name__ == "__main__":
    main()
