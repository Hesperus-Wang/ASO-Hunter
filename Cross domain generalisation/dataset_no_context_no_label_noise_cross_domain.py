import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from rinalmo.data.alphabet import Alphabet


PAD_TOKEN = "<pad>"


def _is_missing_value(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip().lower() in {"", "nan", "none", "null"}


def parse_list_cell(x: Any) -> List[str]:
    """Parse list-like cells such as "['cEt', 'DNA']" into a Python list."""
    if _is_missing_value(x):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x]
    if isinstance(x, tuple):
        return [str(v).strip() for v in x]
    s = str(x).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(v).strip() for v in parsed]
        return [str(parsed).strip()]
    except Exception:
        return [s]


def canonicalize_mod_token(token: Any, kind: str) -> str:
    """Normalize common spelling/case variants while keeping true new tokens."""
    if _is_missing_value(token):
        return PAD_TOKEN
    raw = str(token).strip()
    low = raw.lower()
    if low in {"<pad>", "<pad", "pad", "<padding>", "padding"}:
        return PAD_TOKEN

    if kind == "sugar":
        sugar_map = {
            "dna": "DNA",
            "moe": "MOE",
            "2'-moe": "MOE",
            "2-moe": "MOE",
            "cet": "cEt",
            "c-et": "cEt",
            "cet-modified": "cEt",
            "lna": "LNA",
            "locked nucleic acid": "LNA",
            "amino": "amino",
            "m": "m",
        }
        return sugar_map.get(low, raw)

    if kind == "backbone":
        bb_map = {
            "po": "PO",
            "ps": "PS",
            "mop": "MOP",
        }
        return bb_map.get(low, raw.upper() if raw.isalpha() else raw)

    return raw


def canonicalize_method_token(value: Any) -> str:
    """Canonicalize transfection method labels, including compound labels separated by '|'."""
    if _is_missing_value(value):
        return "Other"

    def _one(part: str) -> str:
        p = re.sub(r"\s+", " ", str(part).strip())
        low = p.lower()
        method_map = {
            "other": "Other",
            "electroporation": "Electroporation",
            "gymnosis": "Gymnosis",
            "free uptake": "free uptake",
            "uptake": "free uptake",
            "lipofection": "Lipofection",
            "lipofectamine 2000 reagent": "Lipofectamine 2000",
            "lipofectamine2000@": "Lipofectamine 2000",
            "lipofectamine 2000": "Lipofectamine 2000",
            "lipofectin reagent": "Lipofectin reagent",
            "cytofectin": "Cytofectin",
        }
        return method_map.get(low, p if p else "Other")

    parts = [_one(p) for p in str(value).split("|")]
    parts = [p for p in parts if p and p != "Other"] or ["Other"]

    priority = {
        "Other": 0,
        "Electroporation": 1,
        "Gymnosis": 2,
        "free uptake": 3,
        "Lipofection": 4,
        "Lipofectamine 2000": 5,
        "Lipofectin reagent": 6,
        "Cytofectin": 7,
    }
    seen = set()
    unique: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    unique.sort(key=lambda x: (priority.get(x, 999), x))
    return " | ".join(unique)


def _read_column_values(path: Path, column: str) -> Iterable[Any]:
    df = pd.read_csv(path, usecols=lambda c: c == column)
    if column not in df.columns:
        return []
    return df[column].tolist()


class ASODataset(Dataset):
    """
    ASO inhibition regression dataset for the no-label-noise cross-domain ablation (no-context compatible).

    The chemistry/backbone/transfection vocabularies can be supplied from an
    external union vocabulary built from both the oligoaltas training/validation
    file and the ASoptimizer external-test file. This prevents external-test
    tokens such as LNA, MOP, or ASoptimizer-specific transfection labels from
    being silently mapped to padding/unknown.
    """

    def __init__(
        self,
        data_path: str,
        alphabet: Alphabet,
        pad_to_max_len: bool = True,
        return_task_id: bool = False,
        task_id_column: str = "patent_id",
        chem_vocab: Optional[Dict[str, int]] = None,
        backbone_vocab: Optional[Dict[str, int]] = None,
        transfection_method_vocab: Optional[Dict[str, int]] = None,
        max_enc_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)

        required_cols = [
            "aso_sequence_5_to_3",
            "sugar_mods",
            "backbone_mods",
            "inhibition_percent",
            "dosage",
            "transfection_method",
            "custom_id",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.data_path}: {missing}")

        if "rna_context" not in self.df.columns:
            self.df["rna_context"] = ""
            print("[dataset] `rna_context` column not found. Using empty placeholder context for all samples.")
        else:
            print("[dataset] `rna_context` column found, but the no-context training script will ignore it.")

        self.df["rna_sequence"] = self.df["aso_sequence_5_to_3"].astype(str).str.replace("T", "U", regex=False)
        self.df["rna_context"] = self.df["rna_context"].fillna("").astype(str)
        self.df = self.df.dropna(subset=["inhibition_percent"]).copy()

        self.df["sugar_mods"] = self.df["sugar_mods"].apply(
            lambda x: [canonicalize_mod_token(v, "sugar") for v in parse_list_cell(x)]
        )
        self.df["backbone_mods"] = self.df["backbone_mods"].apply(
            lambda x: [canonicalize_mod_token(v, "backbone") for v in parse_list_cell(x)]
        )
        self.df["transfection_method"] = self.df["transfection_method"].apply(canonicalize_method_token)

        self.df = self.df.reset_index(drop=True)

        self.chem_vocab = chem_vocab or self.build_chem_vocab_from_frames([self.df])
        self.backbone_vocab = backbone_vocab or self.build_backbone_vocab_from_frames([self.df])
        self.transfection_method_vocab = transfection_method_vocab or self.build_method_vocab_from_frames([self.df])

        if PAD_TOKEN not in self.chem_vocab or self.chem_vocab[PAD_TOKEN] != 0:
            raise ValueError("chem_vocab must contain '<pad>' with index 0")
        if PAD_TOKEN not in self.backbone_vocab or self.backbone_vocab[PAD_TOKEN] != 0:
            raise ValueError("backbone_vocab must contain '<pad>' with index 0")

        print("Using Chemistry Vocabulary:", self.chem_vocab)
        print("Using Backbone Vocabulary:", self.backbone_vocab)
        print("Using Transfection Method Vocabulary:", self.transfection_method_vocab)

        self.median_dosage = float(pd.to_numeric(self.df["dosage"], errors="coerce").median())
        if np.isnan(self.median_dosage):
            self.median_dosage = 0.0

        self.alphabet = alphabet
        self.pad_to_max_len = pad_to_max_len
        self.return_task_id = return_task_id
        self.task_id_column = task_id_column

        if self.pad_to_max_len:
            self.max_enc_seq_len = int(max_enc_seq_len) if max_enc_seq_len is not None else int(self.df["rna_sequence"].str.len().max()) + 2
            self.max_context_len = 2
        else:
            self.max_enc_seq_len = None
            self.max_context_len = 2

        self._assign_task_ids(task_id_column)
        self.task_to_indices: Dict[str, List[int]] = {}
        for i, pid in enumerate(self.df["patent_id"].astype(str).tolist()):
            self.task_to_indices.setdefault(pid, []).append(i)

        print(f"[dataset] {self.data_path.name}: {len(self.df)} samples, {len(self.task_to_indices)} task groups using '{task_id_column}'.")

    @staticmethod
    def extract_patent_id(custom_id: str) -> str:
        if custom_id is None:
            return "UNKNOWN"
        s = str(custom_id).replace("\\", "/")
        filename = s.split("/")[-1]
        if "_table_" in filename:
            return filename.split("_table_")[0]
        if "." in filename:
            return filename.rsplit(".", 1)[0]
        return filename if filename else "UNKNOWN"

    def _assign_task_ids(self, task_id_column: str) -> None:
        col = str(task_id_column or "patent_id")
        if col in {"target_gene_cell_line", "target_cell", "gene_cell"}:
            if "target_gene" not in self.df.columns or "cell_line" not in self.df.columns:
                raise ValueError(f"task_id_column={col!r} requires both target_gene and cell_line columns in {self.data_path}")
            self.df["patent_id"] = "TG=" + self.df["target_gene"].astype(str) + "__CL=" + self.df["cell_line"].astype(str)
        elif col in self.df.columns:
            self.df["patent_id"] = self.df[col].astype(str)
        elif "patent_id" in self.df.columns:
            self.df["patent_id"] = self.df["patent_id"].astype(str)
        elif "target_gene" in self.df.columns and "cell_line" in self.df.columns:
            self.df["patent_id"] = "TG=" + self.df["target_gene"].astype(str) + "__CL=" + self.df["cell_line"].astype(str)
            print(f"[dataset] task_id_column={col!r} not found; fallback to target_gene+cell_line.")
        else:
            self.df["patent_id"] = self.df["custom_id"].astype(str).apply(self.extract_patent_id)
            print(f"[dataset] task_id_column={col!r} not found; fallback to patent_id extracted from custom_id.")

    @staticmethod
    def build_chem_vocab_from_frames(frames: Sequence[pd.DataFrame]) -> Dict[str, int]:
        tokens = set()
        for df in frames:
            if "sugar_mods" not in df.columns:
                continue
            for cell in df["sugar_mods"].tolist():
                for token in parse_list_cell(cell):
                    token = canonicalize_mod_token(token, "sugar")
                    if token != PAD_TOKEN:
                        tokens.add(token)
        vocab: Dict[str, int] = {PAD_TOKEN: 0}
        priority = ["DNA", "MOE", "cEt", "LNA"]
        for token in priority:
            if token in tokens or token == "DNA":
                vocab.setdefault(token, len(vocab))
        for token in sorted(tokens):
            vocab.setdefault(token, len(vocab))
        return vocab

    @staticmethod
    def build_backbone_vocab_from_frames(frames: Sequence[pd.DataFrame]) -> Dict[str, int]:
        tokens = set()
        for df in frames:
            if "backbone_mods" not in df.columns:
                continue
            for cell in df["backbone_mods"].tolist():
                for token in parse_list_cell(cell):
                    token = canonicalize_mod_token(token, "backbone")
                    if token != PAD_TOKEN:
                        tokens.add(token)
        vocab: Dict[str, int] = {PAD_TOKEN: 0}
        for token in ["PO", "PS"]:
            if token in tokens or token in {"PO", "PS"}:
                vocab.setdefault(token, len(vocab))
        for token in sorted(tokens):
            vocab.setdefault(token, len(vocab))
        return vocab

    @staticmethod
    def build_method_vocab_from_frames(frames: Sequence[pd.DataFrame]) -> Dict[str, int]:
        tokens = set()
        for df in frames:
            if "transfection_method" not in df.columns:
                continue
            for cell in df["transfection_method"].tolist():
                tokens.add(canonicalize_method_token(cell))
        vocab: Dict[str, int] = {"Other": 0}
        priority = ["Electroporation", "Gymnosis", "free uptake", "Lipofection", "Lipofectamine 2000", "Lipofectin reagent", "Cytofectin"]
        for token in priority:
            if token in tokens:
                vocab.setdefault(token, len(vocab))
        for token in sorted(tokens):
            vocab.setdefault(token, len(vocab))
        return vocab

    @classmethod
    def build_vocabs_from_csvs(cls, paths: Sequence[Path]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        frames: List[pd.DataFrame] = []
        for path in paths:
            if path is None:
                continue
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Vocabulary source file does not exist: {p}")
            usecols = lambda c: c in {"sugar_mods", "backbone_mods", "transfection_method"}
            frames.append(pd.read_csv(p, usecols=usecols))
        if not frames:
            raise ValueError("No CSV files were provided for vocabulary construction.")
        return (
            cls.build_chem_vocab_from_frames(frames),
            cls.build_backbone_vocab_from_frames(frames),
            cls.build_method_vocab_from_frames(frames),
        )

    @staticmethod
    def compute_max_enc_seq_len(paths: Sequence[Path]) -> int:
        max_len = 0
        for path in paths:
            if path is None:
                continue
            p = Path(path)
            df = pd.read_csv(p, usecols=lambda c: c == "aso_sequence_5_to_3")
            if "aso_sequence_5_to_3" not in df.columns or len(df) == 0:
                continue
            lengths = df["aso_sequence_5_to_3"].dropna().astype(str).str.len()
            if len(lengths):
                max_len = max(max_len, int(lengths.max()))
        if max_len <= 0:
            raise ValueError("Could not infer max ASO sequence length from provided CSV files.")
        return max_len + 2

    def __len__(self) -> int:
        return len(self.df)

    def _tokenize_and_pad_mods(self, mods_list: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
        pad_idx = vocab[PAD_TOKEN]
        padded_tokens: List[int] = [pad_idx]  # align with CLS
        tokens = [vocab.get(mod, pad_idx) for mod in (mods_list or [])]
        padded_tokens.extend(tokens)
        padding_needed = max_len - len(padded_tokens)
        if padding_needed > 0:
            padded_tokens.extend([pad_idx] * padding_needed)
        return padded_tokens[:max_len]

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        df_row = self.df.iloc[idx]

        enc_len = self.max_enc_seq_len
        ctx_len = self.max_context_len
        if enc_len is None:
            enc_len = int(len(df_row["rna_sequence"])) + 2
        if ctx_len is None:
            ctx_len = 2

        seq_encoded = torch.tensor(
            self.alphabet.encode(df_row["rna_sequence"], pad_to_len=enc_len),
            dtype=torch.long,
        )
        chem_encoded = torch.tensor(
            self._tokenize_and_pad_mods(df_row["sugar_mods"], self.chem_vocab, enc_len),
            dtype=torch.long,
        )
        backbone_encoded = torch.tensor(
            self._tokenize_and_pad_mods(df_row["backbone_mods"], self.backbone_vocab, enc_len),
            dtype=torch.long,
        )

        context_encoded = torch.tensor(
            self.alphabet.encode("", pad_to_len=ctx_len),
            dtype=torch.long,
        )
        context_encoded[context_encoded == self.alphabet.unk_idx] = self.alphabet.mask_idx

        inhibition = torch.tensor(float(df_row["inhibition_percent"]), dtype=torch.float32)

        dosage_val = pd.to_numeric(df_row["dosage"], errors="coerce")
        if pd.isna(dosage_val):
            dosage_val = self.median_dosage
        dosage = torch.tensor(float(dosage_val), dtype=torch.float32)

        custom_id = df_row["custom_id"]
        method_key = canonicalize_method_token(df_row["transfection_method"])
        transfection_method_encoded = torch.tensor(
            self.transfection_method_vocab.get(method_key, self.transfection_method_vocab.get("Other", 0)),
            dtype=torch.long,
        )

        if self.return_task_id:
            patent_id = str(df_row["patent_id"])
            return (
                seq_encoded,
                chem_encoded,
                backbone_encoded,
                context_encoded,
                inhibition,
                dosage,
                transfection_method_encoded,
                custom_id,
                patent_id,
            )

        return (
            seq_encoded,
            chem_encoded,
            backbone_encoded,
            context_encoded,
            inhibition,
            dosage,
            transfection_method_encoded,
            custom_id,
        )

    def train_val_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Subset, Subset]:
        rng = np.random.default_rng(int(random_state))
        unique_tasks = self.df["patent_id"].astype(str).unique()
        shuffled_tasks = rng.permutation(unique_tasks)
        n_tasks = len(shuffled_tasks)
        if n_tasks == 0:
            raise ValueError("No task groups are available for train/val splitting.")

        denom = float(train_ratio) + float(val_ratio)
        train_fraction = float(train_ratio) / denom if denom > 0 else 0.8
        if n_tasks == 1:
            train_size = 1
        else:
            train_size = int(round(train_fraction * n_tasks))
            train_size = max(1, min(train_size, n_tasks - 1))

        train_tasks = shuffled_tasks[:train_size]
        val_tasks = shuffled_tasks[train_size:]

        train_indices = self.df[self.df["patent_id"].isin(train_tasks)].index.tolist()
        val_indices = self.df[self.df["patent_id"].isin(val_tasks)].index.tolist()

        print(f"Split by task - Train: {len(train_tasks)} tasks ({len(train_indices)} samples)")
        print(f"Val: {len(val_tasks)} tasks ({len(val_indices)} samples)")
        print("No internal test split is created; external test_data_path is used as the test set.")

        split_df = self.df.copy()
        split_df["split"] = "unassigned"
        split_df.loc[train_indices, "split"] = "train"
        split_df.loc[val_indices, "split"] = "val"
        output_filename = self.data_path.parent / f"{self.data_path.stem}.trainval_withsplit.csv"
        split_df.to_csv(output_filename, index=False)
        print(f"Train/val split assignments saved to: {output_filename}")

        return Subset(self, indices=train_indices), Subset(self, indices=val_indices)

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Subset, Subset, Subset]:
        """Backward-compatible wrapper: returns an empty internal test subset."""
        train_ds, val_ds = self.train_val_split(train_ratio=train_ratio, val_ratio=val_ratio, random_state=random_state)
        empty_test = Subset(self, indices=[])
        return train_ds, val_ds, empty_test
