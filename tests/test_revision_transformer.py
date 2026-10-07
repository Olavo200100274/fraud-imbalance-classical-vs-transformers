"""Independent CPU-only checks for transformer inputs and revision guardrails."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import optuna
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main_transformer import _seed_everything, _train_model, preprocess_for_transformer, _resample_arrays
from missing_values import BafAbsenceToNaN
from models.fttransformer import TabularDataset, build_model
from run_revision_training import is_complete
from ft_run_protocol import open_study, reconstruct_tpe_state, validate_reference
from models.fttransformer import suggest_hyperparams
from save_load import save_run, finalise_run
from evaluation.metrics import compute_all_metrics
from experiment_protocol import PROTOCOL_VERSION


HP = {
    "d_token": 8, "n_blocks": 1, "attention_n_heads": 2,
    "ffn_d_hidden_factor": 2.0, "attention_dropout": 0.0,
    "ffn_dropout": 0.0, "residual_dropout": 0.0,
    "learning_rate": 0.001, "weight_decay": 0.00001, "batch_size": 4,
}


class RevisionTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seed_is_applied_before_model_construction(self):
        _seed_everything(42)
        first = build_model(HP, 2, [2])
        _seed_everything(42)
        second = build_model(HP, 2, [2])
        for name, parameter in first.state_dict().items():
            self.assertTrue(torch.equal(parameter, second.state_dict()[name]))

    def test_unknown_categories_have_their_own_embedding(self):
        model = build_model(HP, 2, [2]).eval()
        inputs = []
        handle = model.cat_embeddings[0].register_forward_pre_hook(
            lambda module, arguments: inputs.append(arguments[0].clone()))
        try:
            model(torch.zeros(3, 2), torch.tensor([[-1], [2], [0]]))
        finally:
            handle.remove()
        self.assertEqual(model.cat_embeddings[0].num_embeddings, 3)
        self.assertEqual(inputs[0].tolist(), [2, 2, 0])

    def test_attention_path_reproduces_evaluation_logits(self):
        _seed_everything(42)
        model = build_model(HP, 2, [2]).eval()
        x_num = torch.tensor([[0.2, 1.0], [-0.3, 0.8]])
        x_cat = torch.tensor([[0], [-1]])
        with torch.no_grad():
            ordinary = model(x_num, x_cat)
            explained, attention = model.forward_with_attention(x_num, x_cat)
        torch.testing.assert_close(ordinary, explained, rtol=1e-5, atol=1e-6)
        self.assertEqual(attention[0].shape, (2, 4, 4))
        torch.testing.assert_close(attention[0].sum(-1), torch.ones(2, 4))

    def test_absence_policy_preserves_legitimate_negative_values(self):
        frame = pd.DataFrame({
            "prev_address_months_count": [-1.0, 4.0],
            "intended_balcon_amount": [-1.2, 5.0],
            "credit_risk_score": [-1.0, -99.0],
            "velocity_24h": [-3.0, 2.0],
        })
        converted = BafAbsenceToNaN().fit_transform(frame)
        self.assertTrue(np.isnan(converted.iloc[0, 0]))
        self.assertTrue(np.isnan(converted.iloc[0, 1]))
        self.assertEqual(converted.credit_risk_score.tolist(), [-1.0, -99.0])
        self.assertEqual(converted.velocity_24h.tolist(), [-3.0, 2.0])

    def test_absence_policy_rejects_reordered_features(self):
        frame = pd.DataFrame({"bank_months_count": [-1.0, 2.0], "income": [0.2, 0.4]})
        transformer = BafAbsenceToNaN().fit(frame)
        with self.assertRaisesRegex(ValueError, "feature order"):
            transformer.transform(frame[["income", "bank_months_count"]])

    def test_transformer_dimension_includes_fitted_missing_indicators(self):
        train = pd.DataFrame({
            "bank_months_count": [-1, 2, 3, 4], "income": [0.1, 0.2, 0.3, 0.4],
            "employment_status": ["A", "A", "B", "B"],
        })
        holdout = pd.DataFrame({
            "bank_months_count": [5], "income": [0.5], "employment_status": ["C"],
        })
        numerical, categorical, other_num, other_cat, cards, dimension, *_ = (
            preprocess_for_transformer(train, holdout, "nan_indicators"))
        self.assertEqual(dimension, 3)
        self.assertEqual(numerical.shape[1], dimension)
        self.assertEqual(other_num.shape[1], dimension)
        self.assertTrue(np.isfinite(numerical).all())
        self.assertEqual(cards, [2])
        self.assertEqual(other_cat.tolist(), [[-1]])

    def test_smotenc_control_is_independent_of_global_pandas_output(self):
        import sklearn
        rng = np.random.RandomState(5)
        numerical = rng.normal(size=(50, 2)).astype(np.float32)
        categorical = np.arange(50).reshape(-1, 1) % 2
        labels = np.r_[np.zeros(40), np.ones(10)]
        diagnostics = []
        with sklearn.config_context(transform_output="pandas"):
            _, codes, resampled_labels = _resample_arrays(
                "smotenc_control", numerical, categorical, labels, 2, "unit_test", diagnostics)
        self.assertEqual(len(resampled_labels), 80)
        self.assertTrue(set(np.unique(codes)) <= {0, 1})
        self.assertEqual(diagnostics[0]["class_counts_after"], {"0": 40, "1": 40})

    def test_requested_scheduler_horizon_is_not_the_selected_epoch(self):
        _seed_everything(42)
        model = build_model(HP, 2, [])
        dataset = TabularDataset(np.arange(24, dtype=np.float32).reshape(12, 2) / 24,
                                 y=np.tile([0, 0, 1], 4))
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        with patch("main_transformer.torch.optim.lr_scheduler.CosineAnnealingLR",
                   wraps=scheduler) as constructor:
            state, score, epoch, scores, labels = _train_model(
                model, loader, loader, HP, torch.device("cpu"), max_epochs=2, patience=2)
        self.assertEqual(constructor.call_args.kwargs["T_max"], 2)
        self.assertIn(epoch, [1, 2])
        self.assertEqual(len(scores), 12)
        self.assertTrue(np.isfinite(scores).all())
        self.assertIsNotNone(state)

    def test_transformer_partial_run_is_not_complete(self):
        with tempfile.TemporaryDirectory(prefix="fraud_ft_marker_") as temporary:
            run = Path(temporary)
            for filename in ("completed.json", "metrics_test.json", "y_test.npy",
                             "y_test_scores.npy", "y_val.npy", "y_val_scores.npy"):
                (run / filename).touch()
            (run / "config.json").write_text(json.dumps({"sample_fraction": None}))
            manifest = {"runs": {"synthetic/fttransformer/none": {"run_dir": str(run)}}}
            self.assertFalse(is_complete(manifest, "synthetic", "fttransformer", "none"))

    def test_tpe_legacy_replay_reproduces_the_next_parameters(self):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        for index in range(12):
            trial = study.ask()
            suggest_hyperparams(trial)
            trial.report(index / 12, 0)
            study.tell(trial, index / 12)
        expected_sampler = study.sampler
        # Reconstruct before consuming the next suggestion from the original RNG.
        metadata = reconstruct_tpe_state(study)
        reconstructed_sampler = study.sampler
        study.sampler = expected_sampler
        first = study.ask()
        expected = suggest_hyperparams(first)
        study.tell(first, state=optuna.trial.TrialState.FAIL)
        # Use a separate copy of the original twelve-trial history.
        replay = optuna.create_study(direction="maximize", sampler=reconstructed_sampler)
        replay.add_trials(study.trials[:12])
        actual = suggest_hyperparams(replay.ask())
        self.assertEqual(expected, actual)
        self.assertEqual(metadata["reconstructed_trials"], 12)

    def test_study_rejects_a_changed_sample_with_the_same_path(self):
        metadata = {"sample_fraction": None, "dev_indices_sha256": "dev",
                    "test_indices_sha256": "test", "feature_columns": ["first"]}
        with tempfile.TemporaryDirectory(prefix="fraud_ft_study_guard_") as temporary:
            study, storage, _ = open_study("synthetic", "raw", metadata, "preserve", 200,
                                           results_root=temporary)
            storage.remove_session()
            storage.engine.dispose()
            changed = {**metadata, "sample_fraction": 0.05}
            with self.assertRaisesRegex(ValueError, "incompatible"):
                open_study("synthetic", "raw", changed, "preserve", 200, results_root=temporary)

    def test_ft_baseline_pin_rejects_changed_policy_and_split(self):
        metadata = {"dev_indices": [1, 2], "test_indices": [3], "split_seed": 42,
                    "dev_indices_sha256": "dev", "test_indices_sha256": "test"}
        config = {"dataset": "ulb_2013", "model": "fttransformer", "strategy": "none",
                  "dataset_hash_sha256": "raw", "train_samples": 2, "test_samples": 1,
                  "split_seed": 42, "protocol_version": PROTOCOL_VERSION,
                  "missing_policy": "preserve", "scheduler_horizon": 200,
                  "data_provenance": {"dev_indices_sha256": "dev", "test_indices_sha256": "test"}}
        validate_reference(config, "ulb_2013", "raw", metadata, "preserve", 200)
        with self.assertRaisesRegex(ValueError, "policies differ"):
            validate_reference(config, "ulb_2013", "raw", metadata, "nan_indicators", 200)
        with self.assertRaisesRegex(ValueError, "DEV.*indices differ"):
            validate_reference(config, "ulb_2013", "raw", {**metadata, "dev_indices_sha256": "changed"},
                               "preserve", 200)

    def test_deferred_completion_requires_transformer_extras(self):
        with tempfile.TemporaryDirectory(prefix="fraud_ft_completion_") as temporary:
            labels, scores, threshold = np.array([0, 1]), np.array([0.1, 0.9]), 0.5
            run = Path(save_run(
                {}, {}, compute_all_metrics(labels, scores, threshold), labels, scores,
                {"dataset": "synthetic", "model": "fttransformer", "strategy": "none",
                 "threshold_exact": threshold}, "fttransformer", dataset="synthetic",
                model_type="torch", results_root=temporary, defer_completion=True))
            self.assertFalse((run / "completed.json").exists())
            self.assertFalse((Path(temporary) / "revision_manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "missing"):
                finalise_run(run, required_artefacts=("validation_model.pt",))
            (run / "validation_model.pt").touch()
            finalise_run(run, required_artefacts=("validation_model.pt",))
            self.assertTrue((run / "completed.json").is_file())


if __name__ == "__main__":
    unittest.main()
