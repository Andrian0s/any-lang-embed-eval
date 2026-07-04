#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amr_foil_pipeline.py
====================
Standalone, runnable port of `AMR_generate_datasets.ipynb`.

It has two stages:

  STAGE 1  --  prep
      Load / transform the *original* sources into the parallel-CSV input format:

        flores200 -> <input_dir>/flores200_devtest_parallel.csv   (read as-is; can
                     optionally be rebuilt from the HF mirror)
        wmt24pp   -> <input_dir>/wmt24pp_parallel.csv  built from google/wmt24pp
                     (one row per segment_id, en_EN + <lang> columns) AND with the
                     six Romansh varieties from ZurichNLP/wmt24pp-rm merged in as
                     5-char rm_* columns (remapped from de_DE-rm-<variety>).
        bouquet   -> pivoted in-memory to wide form (see pivot_bouquet()).

      The Romansh columns are *extra* carry-along columns: the WMT24++ foil
      pipeline only ever reads `en_EN`, so adding rm_* does NOT change behaviour or
      break any downstream step. They use the ORIGINAL 5-char rm_* codes (rm_RG,
      rm_SV, rm_ST, rm_SM, rm_PU, rm_VA) so the embedding notebook's 5-character
      language-discovery rule still finds them. (The published dataset renames these
      to roh_* — see ROH_ALIASES.)

  STAGE 2  --  generate
      Run the AMR foil pipeline (spaCy + amrlib parse -> AMR triples -> transform
      -> AMR-to-text generate -> NLI validation) and write the output CSVs
      (UTF-16) with per-transform splits and ALL_successful / ALL_failed splits.

      * flores200  -> sentence-level pipeline (best-of-all scoring per transform)
      * wmt24pp    -> paragraph-level pipeline (greedy first-valid over sentences)
      * bouquet    -> sentence_level rows use the sentence pipeline,
                      paragraph_level rows use the paragraph pipeline

      It also exports the inputs for 2--ALEE_PRE-CALCULATE-Embeddings.ipynb as
      <output_dir>/datasets/alee_{mt61,f200,bq275}.csv (override with
      --embed-inputs-dir):
        alee_mt61.csv  <- wmt24pp_all_foils.csv        (61 langs incl. rm_*)
        alee_f200.csv  <- flores200_ALL_successful.csv (foil_<t>_eng_Latn)
        alee_bq275.csv <- bouquet_ALL_successful.csv   (+ level column)

Stage `generate` needs the heavy stack (torch, transformers<4.50, amrlib==0.8.0,
spacy en_core_web_sm, penman, nltk, the amrlib gtos model and the NLI model
`juliussteen/DeBERTa-v3-FaithAug`). Stage `prep` only needs pandas + datasets and
is CPU-only.

Usage
-----
    python amr_foil_pipeline.py --stage prep     --dataset all
    python amr_foil_pipeline.py --stage generate --dataset flores
    python amr_foil_pipeline.py --stage all      --dataset wmt24 --limit 50

Deterministic: random.seed(42); NLI threshold 0.8.
"""

import argparse
import json
import os
import random
import re
import shutil
import ssl
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42

# ── Romansh: source config (de_DE-rm-<variety>)  ->  ORIGINAL 5-char rm_* column ─
# The downstream embedding notebook (2--ALEE_PRE-CALCULATE-Embeddings.ipynb) discovers
# languages with a strict 5-character rule (`xx_YY`), so Romansh MUST be exported under
# its original 5-char codes or it would be silently skipped. (Validated: each rm_* code
# matches its ZurichNLP/wmt24pp-rm variety 1:1 on segment_id.)
ROMANSH_REMAP = {
    "de_DE-rm-rumgr": "rm_RG",     # Rumantsch Grischun
    "de_DE-rm-sursilv": "rm_SV",   # Sursilvan
    "de_DE-rm-sutsilv": "rm_ST",   # Sutsilvan
    "de_DE-rm-surmiran": "rm_SM",  # Surmiran
    "de_DE-rm-puter": "rm_PU",     # Puter
    "de_DE-rm-vallader": "rm_VA",  # Vallader
}
# The published dataset (Psychias/alee_datasets) instead exposes these as roh_* names:
ROH_ALIASES = {
    "rm_RG": "roh_rumgr", "rm_SV": "roh_sursilv", "rm_ST": "roh_sutsilv",
    "rm_SM": "roh_surmiran", "rm_PU": "roh_puter", "rm_VA": "roh_vallader",
}

# Foil-output CSV -> embedding-notebook input filename (2--ALEE_PRE-CALCULATE-Embeddings).
EMBED_INPUT_MAP = {
    "flores": ("flores200_ALL_successful.csv", "alee_f200.csv"),
    "wmt24":  ("wmt24pp_all_foils.csv",        "alee_mt61.csv"),
    "bouquet": ("bouquet_ALL_successful.csv",  "alee_bq275.csv"),
}

# WMT24++ target language-pair configs -> wide column name (== the part after "en-")
# Built dynamically from the available configs at prep time.


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class FoilEngine:
    """Holds the heavy models and the foil logic."""

    def __init__(self, device=None):
        import numpy as np
        import torch
        import nltk
        import amrlib
        import spacy
        import penman
        import logging
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from nltk.tokenize import sent_tokenize
        from nltk.corpus import wordnet as wn

        self.np, self.torch, self.penman = np, torch, penman
        self.sent_tokenize, self.wn = sent_tokenize, wn

        logging.getLogger("penman").setLevel(logging.ERROR)
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

        nltk.download("punkt_tab", quiet=True)
        for res in ["punkt", "wordnet", "averaged_perceptron_tagger"]:
            try:
                nltk.data.find(f"tokenizers/{res}") if res == "punkt" else nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)

        random.seed(SEED)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> FoilEngine on {self.device}")

        amrlib.setup_spacy_extension()
        self.nlp = spacy.load("en_core_web_sm")
        self.gtos_model = amrlib.load_gtos_model(device=self.device)

        nli_name = "juliussteen/DeBERTa-v3-FaithAug"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(self.device)
        assert self.nli_model.config.id2label[0] == "entailment", "Entailment must be label 0"

        self.foil_pipelines = {
            "polarity_negation": {"augment": self.polarity_negation, "min_len": 25},
            "role_swap": {"augment": self.random_swap, "min_len": 25},
            "antonym_replacement": {"augment": self.antonym_replacement, "min_len": 25},
            "hypernym_substitution": {"augment": self.hypernym_substitution, "min_len": 25},
        }

    # ---- AMR parse ----
    def sentences_to_graphstrings(self, text):
        return self.nlp(text)._.to_amr()

    # ---- WordNet utilities ----
    def _best_synsets(self, word):
        wn = self.wn
        word_prefix = word.split("-")[0] if "-" in word else word
        best = []
        for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
            s = wn.synsets(word_prefix, pos=pos)
            if len(s) > len(best):
                best = s
        return best

    def get_hypernyms(self, word):
        out = set()
        for syn in self._best_synsets(word):
            if syn.hypernyms():
                for hyper in syn.hypernyms():
                    out.update(hyper.lemma_names())
            else:
                out.update(syn.lemma_names())
        return list(out)

    def get_antonyms(self, word):
        out = set()
        for syn in self._best_synsets(word):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    out.add(ant.name())
        return list(out)

    # ---- AMR graph transforms ----
    @staticmethod
    def find_leaf_nodes_and_edges(triples):
        graph = defaultdict(list)
        for s, r, o in triples:
            if r != ":instance":
                graph[s].append((r, o))
                graph[o].append((r, s))
        return {(node, graph[node][0]) for node in graph if len(graph[node]) == 1}

    def random_swap(self, triples):
        leaf = list(self.find_leaf_nodes_and_edges(triples))
        if len(leaf) < 2:
            return triples
        (n1, e1), (n2, e2) = random.sample(leaf, 2)
        new = []
        for s, r, o in triples:
            if (s, r, o) == (n1, e1[0], e1[1]):
                new.append((n1, e2[0], e2[1]))
            elif (s, r, o) == (e1[1], e1[0], n1):
                new.append((e2[1], e2[0], n1))
            elif (s, r, o) == (n2, e2[0], e2[1]):
                new.append((n2, e1[0], e1[1]))
            elif (s, r, o) == (e2[1], e2[0], n2):
                new.append((e1[1], e1[0], n2))
            else:
                new.append((s, r, o))
        return new

    @staticmethod
    def polarity_negation(triples):
        pron = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        nodes = list(set(s for s, r, o in triples if o not in pron))
        if not nodes:
            return triples
        target = random.choice(nodes)
        return triples + [(target, ":polarity", "-")]

    def antonym_replacement(self, triples):
        pron = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        candidates = [(s, r, o) for s, r, o in triples if r == ":instance" and o not in pron]
        while candidates:
            s, r, o = random.choice(candidates)
            candidates.remove((s, r, o))
            if "-" in o:
                prefix, suffix = o.split("-", 1)
                ants = self.get_antonyms(prefix)
                if ants:
                    new_o = random.choice(ants) + "-" + suffix
                    return [(s, r, new_o) if (s == a and r == b and o == c) else (a, b, c) for a, b, c in triples]
            else:
                ants = self.get_antonyms(o)
                if ants:
                    new_o = random.choice(ants)
                    return [(s, r, new_o) if (s == a and r == b and o == c) else (a, b, c) for a, b, c in triples]
        return triples

    def hypernym_substitution(self, triples):
        pron = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
        candidates = [(s, r, o) for s, r, o in triples if r == ":instance" and o not in pron]
        if not candidates:
            return triples
        s, r, o = random.choice(candidates)
        if "-" in o:
            prefix, suffix = o.split("-", 1)
            hyps = self.get_hypernyms(prefix)
            if hyps:
                new_o = random.choice(hyps) + "-" + suffix
                return [(s, r, new_o) if (s == a and r == b and o == c) else (a, b, c) for a, b, c in triples]
        else:
            hyps = self.get_hypernyms(o)
            if hyps:
                new_o = random.choice(hyps)
                return [(s, r, new_o) if (s == a and r == b and o == c) else (a, b, c) for a, b, c in triples]
        return triples

    # ---- text + NLI ----
    @staticmethod
    def post_process_text(original, foil):
        if not foil:
            return None
        if original and original[0].isupper() and foil:
            foil = foil[0].upper() + foil[1:]
        return re.sub(r"\s+([.,;!?])", r"\1", foil)

    @staticmethod
    def clean_str(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def generate_from_triples(self, mod_triples):
        encoded = self.penman.encode(self.penman.Graph(mod_triples))
        try:
            gen = self.gtos_model.generate([encoded], disable_progress=True)
        except TypeError:
            gen = self.gtos_model.generate([encoded])
        if not gen:
            return None
        return gen[0][0] if isinstance(gen[0], list) else gen[0]

    def get_nli_label_and_score(self, premise, hypothesis):
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            outputs = self.nli_model(**inputs)
        probs = self.torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = self.np.argmax(probs)
        return (label == 0), float(probs[0])

    def validate_foil(self, original, foil, threshold=0.8):
        is_fwd, p_fwd = self.get_nli_label_and_score(original, foil)
        is_bwd, p_bwd = self.get_nli_label_and_score(foil, original)
        info = {"forward": {"prob": p_fwd, "is_entailment": is_fwd},
                "backward": {"prob": p_bwd, "is_entailment": is_bwd}}
        if is_fwd and is_bwd:
            return False, "failed_bidirectional_entailment", info
        if p_fwd > threshold or p_bwd > threshold:
            return False, "failed_high_probability", info
        return True, "success", info

    @staticmethod
    def _score(info):
        f, b = info["forward"]["is_entailment"], info["backward"]["is_entailment"]
        return 2.0 if (f and b) else (1.0 if (f or b) else 0.0)

    # ---- SENTENCE-LEVEL (FLORES / bouquet sentence_level) ----
    def process_sentence_row(self, text):
        best = {k: {"text": text, "status": "no_change", "entailment_probs": None, "priority": 0}
                for k in self.foil_pipelines}
        if pd.isna(text) or not isinstance(text, str) or len(text) < 25:
            return self._format_sentence_output(best)
        try:
            graphs = self.sentences_to_graphstrings(text)
            if not graphs:
                return self._format_sentence_output(best)
            graph_string = graphs[0]
            for name, pipe in self.foil_pipelines.items():
                if len(text) < pipe["min_len"]:
                    continue
                g = self.penman.decode(graph_string)
                mod = pipe["augment"](g.triples)
                if mod == g.triples:
                    continue
                raw = self.generate_from_triples(mod)
                if not raw:
                    continue
                foil = self.post_process_text(text, raw)
                if not foil or self.clean_str(text) == self.clean_str(foil):
                    continue
                is_valid, status, info = self.validate_foil(text, foil)
                cur_prio = 2 if is_valid else 1
                cur_status = "success" if is_valid else status
                cur_score = self._score(info)
                ex = best[name]
                ex_score = self._score(ex["entailment_probs"]) if ex["entailment_probs"] else float("inf")
                if cur_prio > ex["priority"] or (cur_prio == ex["priority"] and cur_score < ex_score):
                    best[name] = {"text": foil, "status": cur_status, "entailment_probs": info, "priority": cur_prio}
        except Exception:
            pass
        return self._format_sentence_output(best)

    @staticmethod
    def _format_sentence_output(candidates):
        out = {}
        for key, data in candidates.items():
            out[f"foil_{key}_eng_Latn"] = data["text"]
            out[f"foil_{key}_status"] = data["status"]
            ep = data["entailment_probs"]
            out[f"foil_{key}_entailment_fwd_prob"] = ep["forward"]["prob"] if ep else None
            out[f"foil_{key}_entailment_bwd_prob"] = ep["backward"]["prob"] if ep else None
            out[f"foil_{key}_is_entailment_fwd"] = ep["forward"]["is_entailment"] if ep else None
            out[f"foil_{key}_is_entailment_bwd"] = ep["backward"]["is_entailment"] if ep else None
        return pd.Series(out)

    # ---- PARAGRAPH-LEVEL (WMT24PP / bouquet paragraph_level) ----
    def process_text_for_single_transformation(self, text, pipe_name, pipe):
        result = {"original_text": text, "foil_text": text, "status": "no_change",
                  "original_sentence": None, "foil_sentence": None,
                  "entailment_fwd_prob": None, "entailment_bwd_prob": None,
                  "is_entailment_fwd": None, "is_entailment_bwd": None}
        if pd.isna(text) or not isinstance(text, str) or len(text) < pipe["min_len"]:
            return result
        sents = self.sent_tokenize(text)
        if len(sents) > 1:
            for sent in sents:
                if len(sent) < 20:
                    continue
                try:
                    graphs = self.sentences_to_graphstrings(sent)
                    if not graphs:
                        continue
                    g = self.penman.decode(graphs[0])
                    mod = pipe["augment"](g.triples)
                    if mod == g.triples:
                        continue
                    raw = self.generate_from_triples(mod)
                    if not raw:
                        continue
                    foil_sent = self.post_process_text(sent, raw)
                    if not foil_sent or self.clean_str(sent) == self.clean_str(foil_sent):
                        continue
                    foil_par = text.replace(sent, foil_sent, 1)
                    is_valid, status, info = self.validate_foil(text, foil_par)
                    if is_valid:
                        self._fill(result, foil_par, status, sent, foil_sent, info)
                        return result
                    elif result["status"] == "no_change":
                        self._fill(result, foil_par, status, sent, foil_sent, info)
                except Exception:
                    pass
        else:
            try:
                graphs = self.sentences_to_graphstrings(text)
                if not graphs:
                    return result
                g = self.penman.decode(graphs[0])
                mod = pipe["augment"](g.triples)
                if mod == g.triples:
                    return result
                raw = self.generate_from_triples(mod)
                if not raw:
                    return result
                foil = self.post_process_text(text, raw)
                if not foil or self.clean_str(text) == self.clean_str(foil):
                    return result
                is_valid, status, info = self.validate_foil(text, foil)
                self._fill(result, foil, status, text, foil, info)
            except Exception:
                pass
        return result

    @staticmethod
    def _fill(result, foil_text, status, orig_sent, foil_sent, info):
        result.update({
            "foil_text": foil_text, "status": status,
            "original_sentence": orig_sent, "foil_sentence": foil_sent,
            "entailment_fwd_prob": info["forward"]["prob"],
            "entailment_bwd_prob": info["backward"]["prob"],
            "is_entailment_fwd": info["forward"]["is_entailment"],
            "is_entailment_bwd": info["backward"]["is_entailment"],
        })

    def process_paragraph_row(self, text):
        row = {}
        for name, pipe in self.foil_pipelines.items():
            r = self.process_text_for_single_transformation(text, name, pipe)
            row[f"foil_{name}_eng_Latn"] = r["foil_text"]
            row[f"foil_{name}_status"] = r["status"]
            row[f"foil_{name}_original_sentence"] = r["original_sentence"]
            row[f"foil_{name}_foil_sentence"] = r["foil_sentence"]
            row[f"foil_{name}_entailment_fwd_prob"] = r["entailment_fwd_prob"]
            row[f"foil_{name}_entailment_bwd_prob"] = r["entailment_bwd_prob"]
            row[f"foil_{name}_is_entailment_fwd"] = r["is_entailment_fwd"]
            row[f"foil_{name}_is_entailment_bwd"] = r["is_entailment_bwd"]
        return row


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 -- prep  (build the parallel input files)
# ══════════════════════════════════════════════════════════════════════════════
def _read_csv_any(path):
    try:
        return pd.read_csv(path, encoding="utf-16")
    except (UnicodeError, UnicodeDecodeError):
        return pd.read_csv(path)


def prep_flores(input_dir, out_dir, rebuild_from_hf=False):
    """FLORES devtest parallel CSV."""
    dst = out_dir / "flores200_devtest_parallel.csv"
    existing = input_dir / "flores200_devtest_parallel.csv"
    if not rebuild_from_hf and existing.exists():
        df = _read_csv_any(existing)
        df.to_csv(dst, index=False, encoding="utf-16")
        print(f"[flores] reused {existing.name}: {df.shape} -> {dst}")
        return dst
    # Rebuild from the HF mirror (requires the loader script / trust_remote_code).
    from datasets import load_dataset
    print("[flores] rebuilding devtest parallel from Muennighoff/flores200 ...")
    ds = load_dataset("Muennighoff/flores200", "all", split="devtest", trust_remote_code=True)
    df = ds.to_pandas()
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(dst, index=False, encoding="utf-16")
    print(f"[flores] built: {df.shape} -> {dst}")
    return dst


def prep_wmt24(input_dir, out_dir, build_base_from_hf=False):
    """WMT24++ parallel CSV (en_EN + <lang>) with Romansh rm_* merged in.

    The base wide table is either reused from <input_dir>/wmt24pp_parallel.csv or
    rebuilt from google/wmt24pp. Romansh is always merged fresh from
    ZurichNLP/wmt24pp-rm and remapped de_DE-rm-<variety> -> rm_<CODE> (5-char, so the
    downstream embedding notebook discovers it as a language).
    """
    from datasets import load_dataset, get_dataset_config_names

    dst = out_dir / "wmt24pp_parallel.csv"
    base_path = input_dir / "wmt24pp_parallel.csv"

    if not build_base_from_hf and base_path.exists():
        base = _read_csv_any(base_path)
        print(f"[wmt24] reused base {base_path.name}: {base.shape}")
    else:
        print("[wmt24] building base wide table from google/wmt24pp ...")
        configs = [c for c in get_dataset_config_names("google/wmt24pp") if c.startswith("en-")]
        base = None
        for cfg in sorted(configs):
            lang = cfg.split("en-", 1)[1]  # e.g. de_DE
            d = load_dataset("google/wmt24pp", cfg, split="train").to_pandas()
            meta_cols = ["domain", "document_id", "segment_id", "is_bad_source"]
            if base is None:
                base = d[meta_cols + ["source"]].rename(columns={"source": "en_EN"}).copy()
            part = d[["segment_id", "target"]].rename(columns={"target": lang})
            base = base.merge(part, on="segment_id", how="left")
        print(f"[wmt24] built base: {base.shape}")

    # ── merge Romansh (remap de_DE-rm-<variety> -> rm_<CODE>, 5-char) ──
    print("[wmt24] merging Romansh from ZurichNLP/wmt24pp-rm ...")
    for cfg, col in ROMANSH_REMAP.items():
        d = load_dataset("ZurichNLP/wmt24pp-rm", cfg, split="test").to_pandas()
        part = d[["segment_id", "target"]].rename(columns={"target": col})
        base = base.merge(part, on="segment_id", how="left")
    base.to_csv(dst, index=False, encoding="utf-16")
    print(f"[wmt24] wrote {base.shape} (incl. {list(ROMANSH_REMAP.values())}) -> {dst}")
    return dst


def pivot_bouquet(df_raw, level_label):
    """Pivot one BOUQuET level to wide form (one row per uniq_id)."""
    df_eng = df_raw[["uniq_id", "tgt_text"]].drop_duplicates(subset="uniq_id")
    df_eng = df_eng.rename(columns={"tgt_text": "sentence_eng_Latn"})
    df_src = df_raw[df_raw["src_lang"] != "eng_Latn"][["uniq_id", "src_lang", "src_text"]]
    df_src = df_src.drop_duplicates(subset=["uniq_id", "src_lang"])
    df_src_wide = df_src.pivot(index="uniq_id", columns="src_lang", values="src_text").reset_index()
    df_src_wide.columns = [f"sentence_{c}" if c != "uniq_id" else c for c in df_src_wide.columns]
    df = df_eng.merge(df_src_wide, on="uniq_id", how="left")
    meta_cols = ["uniq_id", "domain", "register", "tags", "level", "split", "par_id",
                 "par_comment", "sent_comment", "orig_text", "has_hashtag", "has_emoji",
                 "has_12p", "has_speaker_tag", "newline_next"]
    present = [c for c in meta_cols if c in df_raw.columns]
    df = df.merge(df_raw[present].drop_duplicates(subset="uniq_id"), on="uniq_id", how="left")
    df["level"] = level_label
    return df


def prep_bouquet(out_dir):
    """Pivot facebook/bouquet (test) to wide form.
    Returns (df_sent, df_para) and also writes them for inspection."""
    from datasets import load_dataset
    from huggingface_hub import login
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok)
    print("[bouquet] loading facebook/bouquet (test) ...")
    df_sent = pivot_bouquet(load_dataset("facebook/bouquet", "sentence_level", split="test").to_pandas(),
                            "sentence_level")
    df_para = pivot_bouquet(load_dataset("facebook/bouquet", "paragraph_level", split="test").to_pandas(),
                            "paragraph_level")
    df_sent.to_csv(out_dir / "bouquet_sentence_level_input.csv", index=False, encoding="utf-16")
    df_para.to_csv(out_dir / "bouquet_paragraph_level_input.csv", index=False, encoding="utf-16")
    print(f"[bouquet] sentence: {df_sent.shape}  paragraph: {df_para.shape}")
    return df_sent, df_para


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 -- generate  (run the foil pipeline and write the output CSVs)
# ══════════════════════════════════════════════════════════════════════════════
def _reorder_flores_like(df_final, foil_keys):
    id_col = ["id"]
    text_cols = [c for c in df_final.columns if c.startswith("foil_") and c.endswith("_eng_Latn")]
    status_cols = [c for c in df_final.columns if c.startswith("foil_") and c.endswith("_status")]
    prob_cols = [c for c in df_final.columns if c.startswith("foil_") and "_entailment_" in c]
    other = [c for c in df_final.columns if c not in id_col + text_cols + status_cols + prob_cols]
    return df_final[id_col + text_cols + status_cols + prob_cols + other]


def _split_and_save(df_final, foil_keys, out_dir, prefix, base_cols):
    from tqdm import tqdm
    for transform in foil_keys:
        status_col = f"foil_{transform}_status"
        if status_col not in df_final.columns:
            continue
        cols = [c for c in base_cols if c in df_final.columns] + \
               [c for c in df_final.columns if c.startswith(f"foil_{transform}_")]
        succ = df_final[df_final[status_col] == "success"][cols]
        fail = df_final[df_final[status_col].str.startswith("failed", na=False)][cols]
        if len(succ):
            succ.to_csv(out_dir / f"{prefix}_{transform}_successful.csv", index=False, encoding="utf-16")
        if len(fail):
            fail.to_csv(out_dir / f"{prefix}_{transform}_failed.csv", index=False, encoding="utf-16")
    overall = pd.Series(False, index=df_final.index)
    for transform in foil_keys:
        sc = f"foil_{transform}_status"
        if sc in df_final.columns:
            overall |= (df_final[sc] == "success")
    df_final[overall].to_csv(out_dir / f"{prefix}_ALL_successful.csv", index=False, encoding="utf-16")
    df_final[~overall].to_csv(out_dir / f"{prefix}_ALL_failed.csv", index=False, encoding="utf-16")
    print(f"[{prefix}] ALL_successful={int(overall.sum())}  ALL_failed={int((~overall).sum())}")


def generate_flores(engine, input_csv, out_dir, limit=None):
    from tqdm import tqdm
    tqdm.pandas()
    df = _read_csv_any(input_csv)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if limit:
        df = df.head(limit).copy()
    foils = df["sentence_eng_Latn"].progress_apply(engine.process_sentence_row)
    df_final = _reorder_flores_like(pd.concat([df, foils], axis=1), engine.foil_pipelines.keys())
    df_final.to_csv(out_dir / "flores200_all_foils.csv", index=False, encoding="utf-16")
    _split_and_save(df_final, list(engine.foil_pipelines), out_dir, "flores200",
                    base_cols=["id", "sentence_eng_Latn"])


def generate_wmt24(engine, input_csv, out_dir, limit=None):
    from tqdm import tqdm
    df = _read_csv_any(input_csv)
    if limit:
        df = df.head(limit).copy()
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="wmt24 rows"):
        res = engine.process_paragraph_row(row["en_EN"])
        # the wmt24 foil text column is named 'foil_<t>_text'
        res = {(k.replace("_eng_Latn", "_text")): v for k, v in res.items()}
        res["en_EN"] = row["en_EN"]
        results.append(res)
    df_foils = pd.DataFrame(results)
    df_final = pd.concat([df.reset_index(drop=True),
                          df_foils.drop(columns=["en_EN"])], axis=1)
    # WMT24 output set:
    #   all_foils + per-transform _all + combined ALL_successful / ALL_failed
    df_final.to_csv(out_dir / "wmt24pp_all_foils.csv", index=False, encoding="utf-16")
    for transform in engine.foil_pipelines:
        cols = ["en_EN"] + [c for c in df_final.columns if c.startswith(f"foil_{transform}_")]
        df_final[cols].to_csv(out_dir / f"wmt24pp_{transform}_all.csv", index=False, encoding="utf-16")
    overall = pd.Series(False, index=df_final.index)
    for transform in engine.foil_pipelines:
        overall |= (df_final[f"foil_{transform}_status"] == "success")
    df_final[overall].to_csv(out_dir / "wmt24pp_ALL_successful.csv", index=False, encoding="utf-16")
    df_final[~overall].to_csv(out_dir / "wmt24pp_ALL_failed.csv", index=False, encoding="utf-16")
    print(f"[wmt24pp] ALL_successful={int(overall.sum())}  ALL_failed={int((~overall).sum())}")


def generate_bouquet(engine, df_sent, df_para, out_dir, limit=None):
    from tqdm import tqdm
    if limit:
        df_sent, df_para = df_sent.head(limit).copy(), df_para.head(limit).copy()
    foils_sent = df_sent["sentence_eng_Latn"].progress_apply(engine.process_sentence_row) \
        if hasattr(df_sent["sentence_eng_Latn"], "progress_apply") else \
        df_sent["sentence_eng_Latn"].apply(engine.process_sentence_row)
    df_sent_final = pd.concat([df_sent, foils_sent], axis=1)
    para_results = [engine.process_paragraph_row(r["sentence_eng_Latn"])
                    for _, r in tqdm(df_para.iterrows(), total=len(df_para), desc="bouquet para")]
    df_para_final = pd.concat([df_para, pd.DataFrame(para_results, index=df_para.index)], axis=1)
    for col in df_para_final.columns:
        if col not in df_sent_final.columns:
            df_sent_final[col] = None
    for col in df_sent_final.columns:
        if col not in df_para_final.columns:
            df_para_final[col] = None
    df_final = pd.concat([df_sent_final, df_para_final], ignore_index=True)
    df_final.insert(0, "id", range(1, len(df_final) + 1))
    df_final.to_csv(out_dir / "bouquet_all_foils.csv", index=False, encoding="utf-16")
    df_final[df_final["level"] == "sentence_level"].to_csv(
        out_dir / "bouquet_sentence_level_foils.csv", index=False, encoding="utf-16")
    df_final[df_final["level"] == "paragraph_level"].to_csv(
        out_dir / "bouquet_paragraph_level_foils.csv", index=False, encoding="utf-16")
    _split_and_save(df_final, list(engine.foil_pipelines), out_dir, "bouquet",
                    base_cols=["id", "uniq_id", "level", "sentence_eng_Latn"])


def export_embedding_inputs(out_dir, embed_dir, datasets):
    """Copy the foil-output CSVs to the names the embedding notebook reads
    (2--ALEE_PRE-CALCULATE-Embeddings.ipynb): datasets/alee_{mt61,f200,bq275}.csv.
    Format is preserved verbatim (UTF-16); Romansh stays as 5-char rm_* codes."""
    embed_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        srcname, dstname = EMBED_INPUT_MAP[ds]
        src = out_dir / srcname
        if src.exists():
            shutil.copyfile(src, embed_dir / dstname)
            print(f"[embed-input] {ds}: {src.name} -> {embed_dir / dstname}")
        else:
            print(f"[embed-input] {ds}: SKIP (missing {src})")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["prep", "generate", "all"], default="all")
    ap.add_argument("--dataset", choices=["flores", "wmt24", "bouquet", "all"], default="all")
    ap.add_argument("--input-dir", default=str(ROOT / "raw_datasets"))
    ap.add_argument("--output-dir", default=str(ROOT / "pipeline_out"))
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug).")
    ap.add_argument("--rebuild-flores-from-hf", action="store_true")
    ap.add_argument("--build-wmt24-base-from-hf", action="store_true")
    ap.add_argument("--embed-inputs-dir", default=None,
                    help="Also export alee_{mt61,f200,bq275}.csv here for the embedding "
                         "notebook (default: <output-dir>/datasets).")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["flores", "wmt24", "bouquet"] if args.dataset == "all" else [args.dataset]

    # ---- STAGE prep ----
    prepped = {}
    if args.stage in ("prep", "all"):
        if "flores" in datasets:
            prepped["flores"] = prep_flores(input_dir, out_dir, args.rebuild_flores_from_hf)
        if "wmt24" in datasets:
            prepped["wmt24"] = prep_wmt24(input_dir, out_dir, args.build_wmt24_base_from_hf)
        if "bouquet" in datasets:
            prepped["bouquet"] = prep_bouquet(out_dir)

    if args.stage == "prep":
        print("\n>>> prep complete. Inputs are ready in", out_dir)
        return

    # ---- STAGE generate ----
    engine = FoilEngine()
    if "flores" in datasets:
        src = prepped.get("flores") or (out_dir / "flores200_devtest_parallel.csv")
        if not Path(src).exists():
            src = input_dir / "flores200_devtest_parallel.csv"
        generate_flores(engine, src, out_dir, args.limit)
    if "wmt24" in datasets:
        src = prepped.get("wmt24") or (out_dir / "wmt24pp_parallel.csv")
        if not Path(src).exists():
            src = input_dir / "wmt24pp_parallel.csv"
        generate_wmt24(engine, src, out_dir, args.limit)
    if "bouquet" in datasets:
        if "bouquet" in prepped:
            df_sent, df_para = prepped["bouquet"]
        else:
            df_sent, df_para = prep_bouquet(out_dir)
        generate_bouquet(engine, df_sent, df_para, out_dir, args.limit)

    # export the embedding-notebook inputs (alee_{mt61,f200,bq275}.csv)
    embed_dir = Path(args.embed_inputs_dir) if args.embed_inputs_dir else (out_dir / "datasets")
    export_embedding_inputs(out_dir, embed_dir, datasets)

    print("\n>>> generate complete. Outputs in", out_dir)
    print(">>> embedding-notebook inputs in", embed_dir)


if __name__ == "__main__":
    main()
