import unittest

from src.schemas import TrainingTaskCreateRequest, TrainingTaskDetail


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

    def test_training_task_detail_includes_epoch_metrics(self):
        detail = TrainingTaskDetail(
            task_id="task-1",
            status="completed",
            model_name_cn="测试模型",
            model_name_en="test_model",
            created_at="2026-07-09T00:00:00+00:00",
            started_at="2026-07-09T00:00:01+00:00",
            completed_at="2026-07-09T00:01:00+00:00",
            updated_at="2026-07-09T00:01:00+00:00",
            progress={
                "current_epoch": 2,
                "total_epochs": 2,
                "progress_percentage": 100.0,
                "train_accuracy": 0.9,
                "train_loss": 0.1,
                "val_accuracy": 0.8,
                "val_loss": 0.2,
                "f1_score": 0.75,
            },
            epoch_metrics=[
                {
                    "epoch": 1,
                    "total_epochs": 2,
                    "train_accuracy": 0.7,
                    "train_loss": 0.3,
                    "val_accuracy": 0.6,
                    "val_loss": 0.4,
                    "f1_score": 0.55,
                    "progress_percentage": 50.0,
                },
                {
                    "epoch": 2,
                    "total_epochs": 2,
                    "train_accuracy": 0.9,
                    "train_loss": 0.1,
                    "val_accuracy": 0.8,
                    "val_loss": 0.2,
                    "f1_score": 0.75,
                    "progress_percentage": 100.0,
                },
            ],
            error_message=None,
            artifacts=None,
        )

        self.assertEqual(len(detail.epoch_metrics), 2)
        self.assertEqual(detail.progress.train_accuracy, 0.9)
        self.assertEqual(detail.progress.f1_score, 0.75)
        self.assertEqual(detail.epoch_metrics[-1].epoch, 2)
        self.assertEqual(detail.epoch_metrics[-1].val_loss, 0.2)


if __name__ == "__main__":
    unittest.main()
