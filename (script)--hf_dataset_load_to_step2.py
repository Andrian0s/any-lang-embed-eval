#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hf_to_csv.py
============
Download the published `Psychias/alee_datasets` and convert each config into
the CSV layout read by the embeddings notebook
(`2--ALEE_PRE-CALCULATE-Embeddings.ipynb`):

    datasets/alee_f200.csv    datasets/alee_mt61.csv    datasets/alee_bq275.csv

Output layout:
  * UTF-16 encoded CSVs.
  * f200/bq275: language columns prefixed `sentence_<lang>`; foil columns
    `foil_<type>_eng_Latn` + `foil_<type>_status`; bq275 keeps `level`.
  * mt61: 5-char language codes (`en_EN`, `de_DE`, ...); Romansh under `rm_*`
    (not the published `roh_*`); foil columns `foil_<type>_text` +
    `foil_<type>_status`; keeps `is_bad_source`.
  * A foil cell is "success" iff the published `*_negative` cell is non-null;
    otherwise the original English text is kept with status `no_change`
    (only foils whose status is `success` are embedded downstream).

To regenerate the foils from scratch instead of reusing the published ones,
run `AMR_generate_datasets.ipynb`, which writes the same three files.

Usage:
    python hf_to_csv.py                       # all 3 configs -> ./datasets/
    python hf_to_csv.py --config alee_mt61    # one config
    python hf_to_csv.py --out /content/AMR/datasets
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DEFAULT_REPO = "Psychias/alee_datasets"
DEFAULT_OUT = Path(__file__).resolve().parent / "datasets"

# published negative-column name -> foil pipeline name (order matters for layout)
NEG_TO_FOIL = {
    "PolarityNegation": "polarity_negation",
    "RoleSwap": "role_swap",
    "AntonymRepl": "antonym_replacement",
    "HypernymSub": "hypernym_substitution",
}

# published roh_* -> original 5-char rm_* codes (the embeddings notebook discovers
# languages by a 5-character `xx_XX` rule: `roh_puter` would be skipped, `rm_PU` is found)
ROH_TO_RM = {"roh_rumgr": "rm_RG", "roh_sursilv": "rm_SV", "roh_sutsilv": "rm_ST",
             "roh_surmiran": "rm_SM", "roh_puter": "rm_PU", "roh_vallader": "rm_VA"}

# metadata (non-language) columns per config, as published (build_hf_datasets.py)
META = {
    "alee_f200": ["id", "URL", "domain", "topic", "has_image", "has_hyperlink", "SIB_CATEGORY"],
    "alee_bq275": ["id", "uniq_id", "domain", "register", "tags", "level", "split",
                   "par_id", "par_comment", "orig_text", "newline_next"],
    "alee_mt61": ["domain", "document_id", "segment_id", "is_bad_source"],
}


def _foils_from_negatives(df, eng_prefix, english_col, text_suffix):
    """Rebuild `foil_<type>_<suffix>` + `foil_<type>_status` from `*_negative` cols."""
    out = {}
    for neg_name, foil in NEG_TO_FOIL.items():
        neg_col = f"{eng_prefix}_{neg_name}_negative"
        if neg_col not in df.columns:
            raise KeyError(f"expected column {neg_col!r} not in published data")
        ok = df[neg_col].notna()
        out[f"foil_{foil}_{text_suffix}"] = df[neg_col].where(ok, df[english_col])
        out[f"foil_{foil}_status"] = ok.map({True: "success", False: "no_change"})
    return pd.DataFrame(out, index=df.index)


def _split_negatives(df):
    negs = [c for c in df.columns if c.endswith("_negative")]
    return df.drop(columns=negs), negs


def convert_sentence_config(df, config):
    """f200 / bq275: sentence_<lang> columns + foil_<type>_eng_Latn foils."""
    foils = _foils_from_negatives(df, "eng", "eng_Latn", "eng_Latn")
    base, _ = _split_negatives(df)
    meta = [c for c in META[config] if c in base.columns]
    base = base.rename(columns={c: f"sentence_{c}" for c in base.columns if c not in meta})
    lead = [m for m in ("id",) if m in base.columns]
    rest = [c for c in base.columns if c not in lead]
    return pd.concat([base[lead], foils, base[rest]], axis=1)


def convert_mt61(df):
    """mt61: 5-char language codes (roh_* -> rm_*) + foil_<type>_text foils."""
    foils = _foils_from_negatives(df, "en", "en_EN", "text")
    base, _ = _split_negatives(df)
    base = base.rename(columns=ROH_TO_RM)
    return pd.concat([base, foils], axis=1)


CONVERTERS = {
    "alee_f200": lambda df: convert_sentence_config(df, "alee_f200"),
    "alee_mt61": convert_mt61,
    "alee_bq275": lambda df: convert_sentence_config(df, "alee_bq275"),
}


def build(repo, configs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        df = load_dataset(repo, cfg, split="test").to_pandas()
        result = CONVERTERS[cfg](df)
        path = out_dir / f"{cfg}.csv"
        result.to_csv(path, index=False, encoding="utf-16")
        n_ok = {f: int((result[f"foil_{f}_status"] == "success").sum()) for f in NEG_TO_FOIL.values()}
        print(f">>> {cfg}: {result.shape[0]} rows x {result.shape[1]} cols -> {path}")
        print(f"    successful foils: {n_ok}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repo (default: {DEFAULT_REPO})")
    ap.add_argument("--config", choices=sorted(CONVERTERS), help="only this config (default: all three)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory (default: ./datasets)")
    args = ap.parse_args()

    configs = [args.config] if args.config else list(CONVERTERS)
    build(args.repo, configs, args.out)
    print("done")


if __name__ == "__main__":
    main()
