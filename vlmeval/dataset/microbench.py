import os
import json, re, string, unicodedata
import string
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import load

def _letters(n: int):
    assert 1 <= n <= 26, "This template supports up to 26 options."
    return list(string.ascii_uppercase[:n])  # ['A', ... 'Z'][:n]

def _parse_letter(text: str, num_opts: int) -> str:
    """Strict A..(A+num_opts-1) parser; returns 'INVALID' if no single-letter answer is found."""
    if not isinstance(text, str):
        return "INVALID"
    t = text.strip().upper()
    valid = set(_letters(num_opts))
    if len(t) == 1 and t in valid:
        return t
    # tolerate explanations like "Answer: H ..."
    for ch in reversed(t):
        if ch in valid:
            return ch
    return "INVALID"

class MICROBENCH(ImageBaseDataset):
    """
    Minimal TSV-backed MCQ dataset for VLMEvalKit.

    TSV columns (required):
      - index (int)
      - image_path (str)          # single path or multiple paths joined by ';'
      - question (str)
      - options (JSON list[str])  # e.g., ["None of the above", "Actin", ...]
      - answer (str)              # single letter "A".."Z"

    Optional:
      - hint (str)
      - category (str)
      - split (str)

    Notes:
    - If image_path contains multiple paths separated by ';', we will emit multiple image messages.
    - We keep prompting very strict: model must output a single letter.
    """
    TYPE = 'MCQ'
    NAME = 'MICROBENCH_TSV'
    ALIASES: Sequence[str] = (NAME, 'MICROBENCH')
    DATASET_URL = {NAME: ''}

    # data_file is a path RELATIVE to $LMUData (default: ~/LMUData)
    # Example: data_file='microbench/uBench_classification_10.tsv'
    def __init__(self, dataset: str = NAME, data_file: Optional[str] = None, **kwargs):
        if data_file is None:
            raise ValueError("MICROBENCH requires --data-file/--data_file to point to the exported TSV.")

        self._raw_data_file = data_file
        super().__init__(dataset=dataset, **kwargs)

        required = ['index', 'image_path', 'question', 'options', 'answer']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise KeyError(f"TSV missing required columns: {missing}")

    def load_data(self, dataset):
        tsv_path = self._raw_data_file
        if not os.path.isabs(tsv_path):
            lmu_root = os.environ.get('LMUData', os.path.expanduser('~/MBMU-eval/LMUData'))
            tsv_path = os.path.join(lmu_root, tsv_path)

        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"MICROBENCH TSV not found at {tsv_path}")

        self.data_file = tsv_path
        return pd.read_csv(tsv_path, sep='\t')

    @classmethod
    def supported_datasets(cls):
        return list(cls.ALIASES)

    def build_prompt(self, line):
        """Return VLMEvalKit multimodal messages: [{'type': 'image'|'text', 'value': ...}, ...]"""
        row = self.data.iloc[line] if isinstance(line, int) else line

        # Build image message(s)
        img_field = str(row['image_path'])
        img_paths = [p for p in img_field.split(';') if p.strip()]  # support multi-image
        # Keep as absolute if already absolute; else try to resolve relative to LMUData
        lmu_root = os.environ.get('LMUData', os.path.expanduser('~/LMUData'))
        def resolve(p):
            return p if os.path.isabs(p) else os.path.join(lmu_root, p)
        image_msgs = [dict(type='image', value=resolve(p)) for p in img_paths]

        # Options
        opts = row['options']
        if isinstance(opts, str):
            opts = json.loads(opts)
        letters = _letters(len(opts))
        options_txt = "\n".join(f"{L}. {t}" for L, t in zip(letters, opts))

        # # Optional hint for context (if present)
        # hint = str(row['hint']).strip() if 'hint' in self.data.columns and not pd.isna(row['hint']) else ""
        # hint_txt = (hint + "\n") if hint else ""

        prompt = (
            f"{row['question']}\n"
            f"Options:\n{options_txt}\n"
            f"Respond with the letter between ({letters[0]} and {letters[-1]}) corresponding to the answer choice from the options. No explanation. \n"
            # f"Choose the single best answer that answers the question. Respond with ONE capital letter only between ({letters[0]} and {letters[-1]}). \n"
        )
        return [*image_msgs, dict(type='text', value=prompt)]

    # def evaluate(self, eval_file, **kwargs):
    #     """
    #     Expect model predictions saved by VLMEvalKit under eval_file with at least:
    #     - prediction (raw model text output)
    #     - options (copied from TSV or rejoined) for dynamic parser range
    #     - answer (gold letter)
    #     We compute accuracy and invalid rate.
    #     """
    #     df = load(eval_file)

    #     # Make sure we can get num options per-row; fall back to TSV if missing
    #     if 'options' not in df.columns:
    #         # join with original TSV by 'index'
    #         df = df.merge(self.data[['index', 'options']], on='index', how='left', suffixes=('', '_tsv'))

    #     def _num_opts(opt_field):
    #         try:
    #             return len(json.loads(opt_field)) if isinstance(opt_field, str) else len(opt_field)
    #         except Exception:
    #             return 4

    #     num_opts = df['options'].map(_num_opts)

    #     pred = [
    #         _parse_letter(pred_text, n)
    #         for pred_text, n in zip(df['prediction'], num_opts)
    #     ]
    #     gold = [str(a).strip().upper() for a in df['answer']]

    #     pred = np.array(pred, dtype=object)
    #     gold = np.array(gold, dtype=object)
    #     valid_mask = pred != 'INVALID'

    #     correct = (pred == gold)
    #     acc = float(correct.mean()) if len(correct) else float('nan')
    #     invalid_rate = float((~valid_mask).mean()) if len(valid_mask) else float('nan')

    #     bootstrap_iters = int(kwargs.get('bootstrap_iters', 1000))
    #     confidence_level = float(kwargs.get('confidence_level', 0.95))
    #     bootstrap_seed = kwargs.get('bootstrap_seed', 0)
    #     alpha = max(0.0, min(1.0, (1.0 - confidence_level) / 2.0))

    #     if len(correct) >= 2 and bootstrap_iters > 0:
    #         rng = np.random.default_rng(bootstrap_seed)
    #         samples = np.empty(bootstrap_iters, dtype=float)
    #         for i in range(bootstrap_iters):
    #             idx = rng.integers(0, len(correct), size=len(correct))
    #             samples[i] = correct[idx].mean()
    #         lower = float(np.quantile(samples, alpha))
    #         upper = float(np.quantile(samples, 1.0 - alpha))
    #     else:
    #         lower = float('nan')
    #         upper = float('nan')

    #     return {
    #         'accuracy': [acc],
    #         'invalid_rate': [invalid_rate],
    #         'accuracy_ci_lower': [lower],
    #         'accuracy_ci_upper': [upper],
    #     }


    def _safe_json_list(self,x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                pass
        return None

    def _norm_letter(self,s):
        if s is None:
            return None
        s = unicodedata.normalize("NFKC", str(s)).strip().upper()
        return s if s and "A" <= s <= "Z" else None

    def _extract_letter(self, text: str, n_opts: int) -> Optional[str]:
        if text is None:
            return None
        t = unicodedata.normalize("NFKC", str(text)).strip()
        if not t:
            return None

        # Range of allowed letters: A..hi
        hi = chr(ord('A') + max(1, int(n_opts)) - 1)
        # Case-insensitive
        T = t.upper()

        # 1) Prefer the tail *after* the last occurrence of "Answer"
        pos = T.rfind("ANSWER")
        search_zone = T[pos:] if pos != -1 else T[-100:]  # small tail window as fallback

        # strict single-letter match inside the zone
        m = re.search(rf"\b([A-{hi}])\b", search_zone, flags=re.I)
        if m:
            return m.group(1).upper()

        # common formats at the very beginning of the zone: "C.", "C)", "C -"
        m = re.match(rf"^\s*([A-{hi}])[)\].:\-\s]", search_zone, flags=re.I)
        if m:
            return m.group(1).upper()

        # 2) Global fallback if no 'Answer' tag: first standalone letter anywhere near the end
        m = re.search(rf"\b([A-{hi}])\b", T[-160:], flags=re.I)
        if m:
            return m.group(1).upper()

        return None

    def evaluate(self, eval_file, **kwargs):
        df = load(eval_file)  # VLMEvalKit helper -> DataFrame

        # Normalize column names
        if "prediction" not in df.columns:
            for c in ["pred", "response", "output"]:
                if c in df.columns:
                    df = df.rename(columns={c: "prediction"})
                    break
        assert "prediction" in df.columns, "No 'prediction' column in eval file"
        assert "answer" in df.columns, "No 'answer' (GT letter) column in eval file"
        assert "index" in df.columns, "No 'index' column to join options"

        # print("\n[DEBUG] Sample preds vs gold")
        # print(df[['index','answer','options','prediction']].head(10).to_string())

        # Ensure options present (from eval file or merge from original TSV in self.data)
        if ("options" not in df.columns) or df["options"].isna().any():
            assert hasattr(self, "data") and "options" in self.data.columns, "Original TSV (self.data) lacks 'options'"
            df = df.merge(self.data[["index", "options"]], on="index", how="left", suffixes=("", "_tsv"))

        # Parse number of options per row
        opts = df["options"].apply(self._safe_json_list)
        if opts.isna().any():
            raise ValueError("Missing/invalid 'options' for some rows")
        n_opts = opts.apply(len)

        print(df["prediction"])

        # Extract predicted letter and gold letter
        pred_letters = [
            (self._extract_letter(p, n) or "") for p, n in zip(df["prediction"], n_opts)
        ]
        gold_letters = [self._norm_letter(a) or "" for a in df["answer"]]

        print(f'predictions are {df["prediction"]}')
        pred = np.array(pred_letters, dtype=object)
        print(f'prediction array is {pred}')
        gold = np.array(gold_letters, dtype=object)

        # Accuracy (treat empty/invalid as incorrect)
        correct = (pred == gold)
        acc = float(correct.mean()) if len(correct) else float("nan")

        # Bootstrap CI on accuracy
        iters = int(kwargs.get("bootstrap_iters", 1000))
        cl    = float(kwargs.get("confidence_level", 0.95))
        seed  = kwargs.get("bootstrap_seed", 0)
        if len(correct) >= 2 and iters > 0:
            rng = np.random.default_rng(seed)
            samples = np.empty(iters, dtype=float)
            for i in range(iters):
                idx = rng.integers(0, len(correct), size=len(correct))
                samples[i] = correct[idx].mean()
            alpha = (1.0 - cl) / 2.0
            lo = float(np.quantile(samples, alpha))
            hi = float(np.quantile(samples, 1.0 - alpha))
        else:
            lo = float("nan"); hi = float("nan")

        return {
            "accuracy": [acc],
            "accuracy_ci_lo": [lo],
            "accuracy_ci_hi": [hi],
        }