import torch
from torch.utils.data import Dataset, Subset

import pandas as pd
import numpy as np
import ast

from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

from rinalmo.data.alphabet import Alphabet

from gene_sequence_similarity_split import split_dataframe_by_gene_similarity


class ASODataset(Dataset):
    """
    ASO inhibition regression dataset for the no-adaptive-scheduler ablation
    (no-context compatible).

    Required CSV columns:
      - aso_sequence_5_to_3
      - sugar_mods
      - backbone_mods
      - inhibition_percent
      - dosage
      - transfection_method
      - custom_id

    Optional column:
      - rna_context
        If missing, a dummy empty context will be generated so the existing
        episodic training / collate interface stays compatible.
    """

    def __init__(
        self,
        data_path: str,
        alphabet: Alphabet,
        pad_to_max_len: bool = True,
        return_task_id: bool = False,
        split_strategy: str = "gene_similarity",
        gene_sequence_cache: Optional[str] = None,
        gene_sequence_type: str = "refseq_rna",
        gene_similarity_threshold: float = 0.85,
        gene_kmer_size: int = 5,
        gene_fetch_retmax: int = 5,
        gene_max_nt: int = 200000,
        ncbi_email: Optional[str] = None,
        ncbi_api_key: Optional[str] = None,
        allow_gene_fetch_failures: bool = False,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)

        self.split_strategy = str(split_strategy)
        self.gene_sequence_cache = gene_sequence_cache
        self.gene_sequence_type = str(gene_sequence_type)
        self.gene_similarity_threshold = float(gene_similarity_threshold)
        self.gene_kmer_size = int(gene_kmer_size)
        self.gene_fetch_retmax = int(gene_fetch_retmax)
        self.gene_max_nt = int(gene_max_nt)
        self.ncbi_email = ncbi_email
        self.ncbi_api_key = ncbi_api_key
        self.allow_gene_fetch_failures = bool(allow_gene_fetch_failures)

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

        if self.split_strategy == "gene_similarity":
            sim_required_cols = ["target_gene", "cell_line_species"]
            sim_missing = [c for c in sim_required_cols if c not in self.df.columns]
            if sim_missing:
                raise ValueError(
                    "split_strategy='gene_similarity' requires target_gene and "
                    f"cell_line_species columns, but missing: {sim_missing}"
                )
        elif self.split_strategy != "patent_random":
            raise ValueError(
                f"Unknown split_strategy={self.split_strategy!r}; "
                "expected 'patent_random' or 'gene_similarity'."
            )

        # Context is optional. If absent, create an empty placeholder column.
        if "rna_context" not in self.df.columns:
            self.df["rna_context"] = ""
            print("[dataset] `rna_context` column not found. Using empty placeholder context for all samples.")
        else:
            print("[dataset] `rna_context` column found, but the no-context training script will ignore it.")

        # Convert DNA to RNA (T -> U)
        self.df["rna_sequence"] = self.df["aso_sequence_5_to_3"].astype(str).str.replace("T", "U")

        self.df["rna_context"] = self.df["rna_context"].fillna("").astype(str)

        # Filter out rows with missing inhibition values
        self.df = self.df.dropna(subset=["inhibition_percent"]).copy()

        def _parse_list_cell(x) -> List[str]:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                s = x.strip()
                if s == "":
                    return []
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return parsed
                    return [str(parsed)]
                except Exception:
                    return [s]
            return [str(x)]

        self.df["sugar_mods"] = self.df["sugar_mods"].apply(_parse_list_cell)
        self.df["backbone_mods"] = self.df["backbone_mods"].apply(_parse_list_cell)

        self.transfection_method_vocab = {
            "Electroporation": 0,
            "Gymnosis": 1,
            "Other": 2,
            "Lipofection": 3,
        }
        print("Using Transfection Method Vocabulary:", self.transfection_method_vocab)

        self.chem_vocab = {
            "<pad>": 0,
            "DNA": 1,
            "MOE": 2,
            "cET": 3,
        }
        self.backbone_vocab = {
            "<pad>": 0,
            "PO": 1,
            "PS": 2,
        }
        print("Using Chemistry Vocabulary:", self.chem_vocab)
        print("Using Backbone Vocabulary:", self.backbone_vocab)

        self.df = self.df.reset_index(drop=True)

        self.median_dosage = self.df["dosage"].median()

        self.alphabet = alphabet
        self.pad_to_max_len = pad_to_max_len
        self.return_task_id = return_task_id

        if self.pad_to_max_len:
            self.max_enc_seq_len = int(self.df["rna_sequence"].str.len().max()) + 2
            # Dummy context only needs CLS + EOS (or equivalent special-token length).
            self.max_context_len = 2
        else:
            self.max_enc_seq_len = None
            self.max_context_len = 2

        if "patent_id" not in self.df.columns:
            self.df["patent_id"] = self.df["custom_id"].astype(str).apply(self.extract_patent_id)

        self.task_to_indices: Dict[str, List[int]] = {}
        for i, pid in enumerate(self.df["patent_id"].astype(str).tolist()):
            self.task_to_indices.setdefault(pid, []).append(i)

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

    def __len__(self) -> int:
        return len(self.df)

    def _tokenize_and_pad_mods(self, mods_list: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
        pad_idx = vocab["<pad>"]
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

        # Dummy placeholder context to preserve tuple shape / collate compatibility.
        context_encoded = torch.tensor(
            self.alphabet.encode("", pad_to_len=ctx_len),
            dtype=torch.long,
        )
        context_encoded[context_encoded == self.alphabet.unk_idx] = self.alphabet.mask_idx

        inhibition = torch.tensor(float(df_row["inhibition_percent"]), dtype=torch.float32)

        dosage_val = df_row["dosage"] if pd.notna(df_row["dosage"]) else self.median_dosage
        dosage = torch.tensor(float(dosage_val), dtype=torch.float32)

        custom_id = df_row["custom_id"]

        transfection_method_encoded = torch.tensor(
            self.transfection_method_vocab.get(df_row["transfection_method"], 0),
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

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[Subset, Subset, Subset]:
        if "patent_id" not in self.df.columns:
            self.df["patent_id"] = self.df["custom_id"].astype(str).apply(self.extract_patent_id)

        if self.split_strategy == "gene_similarity":
            split_df, report = split_dataframe_by_gene_similarity(
                self.df,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                random_state=random_state,
                patent_col="patent_id",
                gene_col="target_gene",
                species_col="cell_line_species",
                sequence_cache=self.gene_sequence_cache,
                sequence_type=self.gene_sequence_type,
                similarity_threshold=self.gene_similarity_threshold,
                kmer_size=self.gene_kmer_size,
                fetch_retmax=self.gene_fetch_retmax,
                max_nt=self.gene_max_nt,
                ncbi_email=self.ncbi_email,
                ncbi_api_key=self.ncbi_api_key,
                allow_fetch_failures=self.allow_gene_fetch_failures,
                output_dir=self.data_path.parent,
                output_stem=self.data_path.stem,
            )
            # Preserve the parsed-list columns already prepared by this Dataset, while
            # taking only the split labels from the splitter output.
            self.df["split"] = split_df["split"].values

            train_indices = self.df.index[self.df["split"].eq("train")].tolist()
            val_indices = self.df.index[self.df["split"].eq("val")].tolist()
            test_indices = self.df.index[self.df["split"].eq("test")].tolist()

            n_train_patents = self.df.loc[train_indices, "patent_id"].astype(str).nunique()
            n_val_patents = self.df.loc[val_indices, "patent_id"].astype(str).nunique()
            n_test_patents = self.df.loc[test_indices, "patent_id"].astype(str).nunique()
            print(
                f"Split by gene similarity - Train: {n_train_patents} patents ({len(train_indices)} samples)"
            )
            print(f"Val: {n_val_patents} patents ({len(val_indices)} samples)")
            print(f"Test: {n_test_patents} patents ({len(test_indices)} samples)")
            if report.get("warnings"):
                for warning in report["warnings"]:
                    print(f"[gene-split warning] {warning}")
        else:
            np.random.seed(random_state)

            unique_patents = self.df["patent_id"].astype(str).unique()
            n_patents = len(unique_patents)
            shuffled_patents = np.random.permutation(unique_patents)

            train_size = int(train_ratio * n_patents)
            val_size = int(val_ratio * n_patents)

            train_patents = shuffled_patents[:train_size]
            val_patents = shuffled_patents[train_size : train_size + val_size]
            test_patents = shuffled_patents[train_size + val_size :]

            train_indices = self.df[self.df["patent_id"].isin(train_patents)].index.tolist()
            val_indices = self.df[self.df["patent_id"].isin(val_patents)].index.tolist()
            test_indices = self.df[self.df["patent_id"].isin(test_patents)].index.tolist()

            print(f"Split by patent - Train: {len(train_patents)} patents ({len(train_indices)} samples)")
            print(f"Val: {len(val_patents)} patents ({len(val_indices)} samples)")
            print(f"Test: {len(test_patents)} patents ({len(test_indices)} samples)")

            split_df = self.df.copy()
            split_df["split"] = "unassigned"
            split_df.loc[train_indices, "split"] = "train"
            split_df.loc[val_indices, "split"] = "val"
            split_df.loc[test_indices, "split"] = "test"
            self.df["split"] = split_df["split"].values

            output_filename = self.data_path.parent / f"{self.data_path.stem}.withsplit.csv"
            split_df.to_csv(output_filename, index=False)
            print(f"Split assignments saved to: {output_filename}")

        train_ds = Subset(self, indices=train_indices)
        val_ds = Subset(self, indices=val_indices)
        test_ds = Subset(self, indices=test_indices)

        return train_ds, val_ds, test_ds
