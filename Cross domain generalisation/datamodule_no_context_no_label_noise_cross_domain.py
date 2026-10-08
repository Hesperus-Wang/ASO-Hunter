import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Sampler, Subset

try:
    from dataset_no_context_no_label_noise_cross_domain import ASODataset  # type: ignore
except ImportError:
    from rinalmo.data.downstream.aso_meta.dataset import ASODataset  # type: ignore

try:
    from rinalmo.data.alphabet import Alphabet
except Exception:
    from rinalmo.data.alphabet import Alphabet  # type: ignore


class EpisodicBatchSampler(Sampler):
    """Yield one episode (a list of indices) per iteration."""

    def __init__(
        self,
        task_to_subset_indices: Dict[str, List[int]],
        n_support: int,
        n_query: int,
        episodes_per_epoch: int,
        seed: int = 42,
        min_task_size: int = 2,
        sample_with_replacement_if_needed: bool = True,
    ):
        super().__init__()
        self.n_support = int(n_support)
        self.n_query = int(n_query)
        self.k = self.n_support + self.n_query
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.sample_with_replacement_if_needed = bool(sample_with_replacement_if_needed)

        filtered = {
            t: idxs for t, idxs in task_to_subset_indices.items()
            if idxs is not None and len(idxs) >= min_task_size
        }
        if len(filtered) == 0:
            raise ValueError(
                "EpisodicBatchSampler got 0 valid tasks. "
                "Check that the split contains task groups with at least two samples."
            )
        self.task_to_indices = filtered
        self.tasks = list(self.task_to_indices.keys())
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            task = self.rng.choice(self.tasks)
            pool = self.task_to_indices[task]
            if len(pool) >= self.k:
                chosen = self.rng.sample(pool, self.k)
            else:
                if not self.sample_with_replacement_if_needed:
                    raise ValueError(f"Task '{task}' has only {len(pool)} samples, but episode needs {self.k}.")
                chosen = [self.rng.choice(pool) for _ in range(self.k)]
            yield chosen


def episodic_collate(batch: List[Any], n_support: int) -> Dict[str, Dict[str, Any]]:
    if len(batch) <= n_support:
        raise ValueError(f"Episode batch size {len(batch)} must be > n_support={n_support}.")

    sup = batch[:n_support]
    qry = batch[n_support:]

    def _stack(items, i: int) -> torch.Tensor:
        return torch.stack([x[i] for x in items], dim=0)

    def _pack(items):
        return {
            "aso": _stack(items, 0),
            "chem": _stack(items, 1),
            "backbone": _stack(items, 2),
            "context": _stack(items, 3),
            "y": _stack(items, 4).float(),
            "dosage": _stack(items, 5).float(),
            "method": _stack(items, 6).long(),
            "custom_id": [x[7] for x in items],
            "task_id": [x[8] for x in items],
        }

    return {"support": _pack(sup), "query": _pack(qry)}


def _build_task_to_subset_indices(subset: Subset, k_min: int = 2) -> Dict[str, List[int]]:
    base_dataset = subset.dataset
    if not hasattr(base_dataset, "task_to_indices"):
        raise ValueError("Base dataset has no `task_to_indices`.")

    subset_base_indices: List[int] = list(subset.indices)
    base_to_subset_pos = {base_idx: pos for pos, base_idx in enumerate(subset_base_indices)}
    subset_base_set = set(subset_base_indices)

    task_to_subset: Dict[str, List[int]] = {}
    for task_id, base_idxs in base_dataset.task_to_indices.items():
        kept_base = [i for i in base_idxs if i in subset_base_set]
        if len(kept_base) >= k_min:
            task_to_subset[str(task_id)] = [base_to_subset_pos[i] for i in kept_base]
    return task_to_subset


class ASODataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        test_data_path: Optional[Union[Path, str]] = None,
        task_id_column: str = "patent_id",
        test_task_id_column: Optional[str] = None,
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        random_state: int = 42,
        meta: bool = True,
        n_support: int = 16,
        n_query: int = 16,
        episodes_per_epoch: int = 2000,
        meta_val: bool = False,
        val_episodes: int = 200,
        meta_test: bool = False,
        test_episodes: int = 200,
        sample_with_replacement_if_needed: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.test_data_path = Path(test_data_path) if test_data_path is not None else None
        self.task_id_column = task_id_column
        self.test_task_id_column = test_task_id_column or task_id_column
        self.alphabet = alphabet
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.random_state = int(random_state)

        self.meta = bool(meta)
        self.n_support = int(n_support)
        self.n_query = int(n_query)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.meta_val = bool(meta_val)
        self.val_episodes = int(val_episodes)
        self.meta_test = bool(meta_test)
        self.test_episodes = int(test_episodes)
        self.sample_with_replacement_if_needed = bool(sample_with_replacement_if_needed)

        self.train_task_to_subset_indices: Dict[str, List[int]] = {}
        self.val_task_to_subset_indices: Dict[str, List[int]] = {}
        self.test_task_to_subset_indices: Dict[str, List[int]] = {}
        self._max_enc_seq_len: Optional[int] = None
        self.chem_vocab: Dict[str, int] = {}
        self.backbone_vocab: Dict[str, int] = {}
        self.transfection_method_vocab: Dict[str, int] = {}

    def setup(self, stage: Optional[str] = None):
        if self.test_data_path is None:
            raise ValueError(
                "This no-label-noise external-test datamodule requires `test_data_path`. "
                "Use the in-domain dataset as data_path for train/validation and the external dataset as test_data_path."
            )
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training/validation data_path does not exist: {self.data_path}")
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"External test_data_path does not exist: {self.test_data_path}")

        need_task_id = self.meta or self.meta_val or self.meta_test
        vocab_paths = [self.data_path, self.test_data_path]
        self.chem_vocab, self.backbone_vocab, self.transfection_method_vocab = ASODataset.build_vocabs_from_csvs(vocab_paths)
        self._max_enc_seq_len = ASODataset.compute_max_enc_seq_len(vocab_paths)

        print("[DataModule] Built union vocabularies from train/val and external-test CSV files.")
        print("[DataModule] chem_vocab:", self.chem_vocab)
        print("[DataModule] backbone_vocab:", self.backbone_vocab)
        print("[DataModule] transfection_method_vocab:", self.transfection_method_vocab)
        print(f"[DataModule] Shared max_enc_seq_len={self._max_enc_seq_len}")

        dataset = ASODataset(
            self.data_path,
            alphabet=self.alphabet,
            pad_to_max_len=True,
            return_task_id=need_task_id,
            task_id_column=self.task_id_column,
            chem_vocab=self.chem_vocab,
            backbone_vocab=self.backbone_vocab,
            transfection_method_vocab=self.transfection_method_vocab,
            max_enc_seq_len=self._max_enc_seq_len,
        )
        self.train_dataset, self.val_dataset = dataset.train_val_split(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            random_state=self.random_state,
        )

        print(f"[DataModule] Loading EXTERNAL test dataset from: {self.test_data_path}")
        test_dataset = ASODataset(
            self.test_data_path,
            alphabet=self.alphabet,
            pad_to_max_len=True,
            return_task_id=need_task_id,
            task_id_column=self.test_task_id_column,
            chem_vocab=self.chem_vocab,
            backbone_vocab=self.backbone_vocab,
            transfection_method_vocab=self.transfection_method_vocab,
            max_enc_seq_len=self._max_enc_seq_len,
        )
        self.test_dataset = Subset(test_dataset, indices=list(range(len(test_dataset))))

        print(f"Dataset sizes - Train: {len(self.train_dataset)}")
        print(f"Val: {len(self.val_dataset)}, External Test: {len(self.test_dataset)}")

        if self.meta:
            self.train_task_to_subset_indices = _build_task_to_subset_indices(self.train_dataset, k_min=2)
            print(f"[Meta] Train tasks available: {len(self.train_task_to_subset_indices)}")
        if self.meta_val:
            self.val_task_to_subset_indices = _build_task_to_subset_indices(self.val_dataset, k_min=2)
            print(f"[Meta] Val tasks available: {len(self.val_task_to_subset_indices)}")
        if self.meta_test:
            self.test_task_to_subset_indices = _build_task_to_subset_indices(self.test_dataset, k_min=2)
            print(f"[Meta] External test tasks available: {len(self.test_task_to_subset_indices)}")

    def export_vocab_json(self, path: Union[str, Path]) -> None:
        payload = {
            "chem_vocab": self.chem_vocab,
            "backbone_vocab": self.backbone_vocab,
            "transfection_method_vocab": self.transfection_method_vocab,
            "max_enc_seq_len": self._max_enc_seq_len,
            "train_val_data_path": str(self.data_path),
            "external_test_data_path": str(self.test_data_path),
            "task_id_column": self.task_id_column,
            "test_task_id_column": self.test_task_id_column,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def train_dataloader(self):
        if not self.meta:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )
        sampler = EpisodicBatchSampler(
            task_to_subset_indices=self.train_task_to_subset_indices,
            n_support=self.n_support,
            n_query=self.n_query,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.random_state,
            min_task_size=2,
            sample_with_replacement_if_needed=self.sample_with_replacement_if_needed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: episodic_collate(b, n_support=self.n_support),
        )

    def val_dataloader(self):
        if not self.meta_val:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        sampler = EpisodicBatchSampler(
            task_to_subset_indices=self.val_task_to_subset_indices,
            n_support=self.n_support,
            n_query=self.n_query,
            episodes_per_epoch=self.val_episodes,
            seed=self.random_state + 1,
            min_task_size=2,
            sample_with_replacement_if_needed=self.sample_with_replacement_if_needed,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: episodic_collate(b, n_support=self.n_support),
        )

    def test_dataloader(self):
        if not self.meta_test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        sampler = EpisodicBatchSampler(
            task_to_subset_indices=self.test_task_to_subset_indices,
            n_support=self.n_support,
            n_query=self.n_query,
            episodes_per_epoch=self.test_episodes,
            seed=self.random_state + 2,
            min_task_size=2,
            sample_with_replacement_if_needed=self.sample_with_replacement_if_needed,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda b: episodic_collate(b, n_support=self.n_support),
        )
