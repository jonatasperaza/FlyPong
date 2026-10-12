#!/usr/bin/env python3
"""Acrescenta o neurotransmissor previsto (coluna `nt`) a um neurons.parquet.

Fonte: tabela publica do MaleCNS por neuronio,
  gs://flyem-male-cns/<versao>/connectome-data/flat-connectome/
      body-neurotransmitters-male-cns-<versao>.feather
(baixada por HTTPS de storage.googleapis.com; nao precisa de token).
Usa `consensus_nt` (previsao por neuronio combinada com a do tipo celular e
com a verdade de campo, quando existe). Grava tambem `nt_confidence`
(predicted_nt_confidence). As arestas e as outras colunas nao mudam.

Uso:
    python scripts/add_neurotransmitters.py --data-dir connectome_v09_in21 --version v0.9
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
BUCKET = "https://storage.googleapis.com/flyem-male-cns"


def table_url(version: str) -> str:
    return (f"{BUCKET}/{version}/connectome-data/flat-connectome/"
            f"body-neurotransmitters-male-cns-{version}.feather")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--version", default=None,
                    help="v0.9 ou v1.0; padrao: tirado de metadata.json (dataset)")
    ap.add_argument("--cache-dir", default=str(HERE / "runs" / "mcns_nt"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = args.version or "v" + meta.get("dataset", "male-cns:v1.0").split(":v")[-1]
    cache = Path(args.cache_dir) / f"body-neurotransmitters-male-cns-{version}.feather"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[add_neurotransmitters] baixando {table_url(version)}", flush=True)
        urllib.request.urlretrieve(table_url(version), cache)
    nt = pd.read_feather(cache, columns=["body", "consensus_nt", "predicted_nt_confidence"])
    nt = nt.rename(columns={"body": "bodyId", "consensus_nt": "nt",
                            "predicted_nt_confidence": "nt_confidence"}).set_index("bodyId")

    path = data_dir / "neurons.parquet"
    ndf = pd.read_parquet(path).drop(columns=["nt", "nt_confidence"], errors="ignore")
    ndf = ndf.join(nt, on="bodyId")
    missing = int(ndf["nt"].isna().sum())
    ndf["nt"] = ndf["nt"].fillna("unknown").astype(str)
    print(f"[add_neurotransmitters] {len(ndf)} neuronios, {missing} sem previsao")
    print(pd.crosstab(ndf["role"], ndf["nt"]).to_string())
    ndf.to_parquet(path, index=False)
    meta["nt_source"] = table_url(version)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[add_neurotransmitters] gravado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
