import unittest

from src.schemas import TrainingTaskCreateRequest


class SchemaDefaultsTests(unittest.TestCase):
    def test_training_defaults_use_optimized_lora_parameters(self):
        request = TrainingTaskCreateRequest(
            model_name_cn="测试模型",
            model_name_en="test_model",
            training_data_file="train.xlsx",
            hyperparameters={
                "text_column": "文本内容",
                "label_column": "标签列",
            },
        )

        hp = request.hyperparameters
        self.assertEqual(hp.epochs, 6)
        self.assertEqual(hp.batch_size, 8)
        self.assertEqual(hp.random_seed, 42)
        self.assertEqual(hp.train_val_split, 0.1)
        self.assertTrue(hp.stratified_split)
        self.assertTrue(hp.anchor_samples_enabled)
        self.assertEqual(hp.anchor_repeat, 15)
        self.assertEqual(hp.classifier_pooling_strategy, "mean_cls")
        self.assertEqual(hp.output_activation, "none")
        self.assertIsNotNone(hp.lora)
        self.assertTrue(hp.lora.enabled)
        self.assertEqual(hp.lora.r, 16)
        self.assertEqual(hp.lora.lora_alpha, 32)
        self.assertEqual(hp.lora.lora_dropout, 0.1)
        self.assertEqual(hp.lora.target_modules, ["query", "key", "value", "dense"])


if __name__ == "__main__":
    unittest.main()
