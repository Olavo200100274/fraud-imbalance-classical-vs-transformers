"""Small tests for the revised FT input and validation artefact contract."""

import unittest
import numpy as np
import pandas as pd
import torch

from missing_values import BafAbsenceToNaN, numeric_pipeline
from main_transformer import preprocess_for_transformer
from models.fttransformer import build_model


class TransformerRevisionTests(unittest.TestCase):
    def test_only_documented_codes_are_missing(self):
        frame = pd.DataFrame({"prev_address_months_count": [-1, 2],
                              "intended_balcon_amount": [-2.0, 4.0],
                              "credit_risk_score": [-1, -4],
                              "velocity_6h": [-1.0, 2.0]})
        result = BafAbsenceToNaN().fit_transform(frame)
        self.assertTrue(pd.isna(result.iloc[0, 0]))
        self.assertTrue(pd.isna(result.iloc[0, 1]))
        np.testing.assert_array_equal(result.credit_risk_score, frame.credit_risk_score)
        np.testing.assert_array_equal(result.velocity_6h, frame.velocity_6h)

    def test_imputer_uses_training_only(self):
        train = pd.DataFrame({"bank_months_count": [-1, 3, 5]})
        other = pd.DataFrame({"bank_months_count": [999, -1]})
        pipeline = numeric_pipeline("nan_indicators")
        pipeline.fit(train)
        pipeline.transform(other)
        self.assertAlmostEqual(pipeline.named_steps["imputer"].statistics_[0], 4)
        self.assertEqual(pipeline.transform(other).shape, (2, 2))

    def test_transformer_sensitivity_dimension_is_actual_output(self):
        train = pd.DataFrame({"bank_months_count": [-1, 3, 5], "cat": ["A", "B", "A"]})
        output = preprocess_for_transformer(train, train, "nan_indicators")
        self.assertEqual(output[5], 2)
        self.assertEqual(output[0].shape[1], output[5])

    def test_unknown_category_has_reserved_embedding_in_both_paths(self):
        hp = {"d_token": 8, "ffn_d_hidden_factor": 2, "n_blocks": 1,
              "attention_n_heads": 2, "attention_dropout": 0,
              "ffn_dropout": 0, "residual_dropout": 0}
        torch.manual_seed(42)
        model = build_model(hp, 1, [2]).eval()
        captured = []
        hook = model.cat_embeddings[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].clone()))
        x_num = torch.zeros(3, 1)
        x_cat = torch.tensor([[-1], [0], [2]])
        with torch.no_grad():
            normal = model(x_num, x_cat)
            manual, _ = model.forward_with_attention(x_num, x_cat)
        hook.remove()
        np.testing.assert_array_equal(captured[0].numpy(), [2, 0, 2])
        np.testing.assert_array_equal(captured[1].numpy(), [2, 0, 2])
        torch.testing.assert_close(normal, manual, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
