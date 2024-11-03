import unittest
import sys
import os
import pandas as pd
from minGPT.pipeline import minGPT

class TestMinGPTPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = minGPT()
        self.data_file_path = os.path.join("minGPT", "tests", "test_data", "test_data.csv")
        self.model_config = self.pipeline.get_default_model_config()
        self.data_config = self.pipeline.get_default_data_config()
        self.data_config.file_path = self.data_file_path

    def load_model_with_config(self, train_dataset):
        """Helper method to load model with vocab and block sizes."""
        self.model_config.vocab_size = train_dataset.get_vocab_size()
        self.model_config.block_size = train_dataset.get_block_size()
        self.pipeline.load_model(self.model_config)

    def test_data_preprocessing(self):
        data_config = self.pipeline.get_default_data_config()
        data_config.file_path = self.data_file_path
        data_config.block_size = 64

        raw_data = pd.read_csv(self.data_file_path, sep="\t")
        print("Unprocessed Data (First 5 Rows):")
        print(raw_data.head())

        train_dataset, test_dataset = self.pipeline.data_preprocessing(data_config)

        self.assertIsNotNone(train_dataset, "Train dataset should not be None")
        self.assertIsNotNone(test_dataset, "Test dataset should not be None")
        self.assertGreater(len(train_dataset), 0, "Train dataset should contain data")
        self.assertGreater(len(test_dataset), 0, "Test dataset should contain data")

    def test_model_loading(self):
        train_dataset, _ = self.pipeline.data_preprocessing(self.data_config)
        self.load_model_with_config(train_dataset)
        self.assertIsNotNone(self.pipeline.model, "Model should be loaded")

    def test_training(self):
        train_config = self.pipeline.get_default_train_config()
        train_config.max_iters = 10
        train_config.device = 'cpu'

        train_dataset, _ = self.pipeline.data_preprocessing(self.data_config)
        self.load_model_with_config(train_dataset)

        def batch_end_callback(trainer):
            print(f"Batch {trainer.iter_num} ended with loss: {trainer.loss.item() if hasattr(trainer, 'loss') else 'N/A'}")

        train_config.call_back = batch_end_callback
        loss = self.pipeline.train(train_config)
        self.assertIsNotNone(loss, "Training loss should not be None")

    def test_generation(self):
        generate_config = self.pipeline.get_default_generate_config()
        generate_config.num_samples = 5
        generate_config.ckpt_path = os.path.join("minGPT", "ckpts", "sample_checkpoint.pt")

        train_dataset, _ = self.pipeline.data_preprocessing(self.data_config)
        self.load_model_with_config(train_dataset)

        generated_out = self.pipeline.generate(generate_config)
        self.assertIsInstance(generated_out, list, "Generation output should be a list")
        self.assertGreater(len(generated_out), 0, "Generated output should contain elements")
        print("Generated Output:", generated_out)

    def test_evaluation(self):
        train_dataset, _ = self.pipeline.data_preprocessing(self.data_config)
        self.pipeline.df = pd.DataFrame({"mol_smiles": ["C(=O)CSCC(CO[Cu])OC(=O)[Au]"]})

        generate_config = self.pipeline.get_default_generate_config()
        generate_config.num_samples = 5
        generate_config.ckpt_path = os.path.join("minGPT", "ckpts", "sample_checkpoint.pt")

        self.load_model_with_config(train_dataset)
        generated_out = self.pipeline.generate(generate_config)
        self.pipeline.gen_df = pd.DataFrame({"mol_smiles": generated_out})

        results = self.pipeline.evaluate()
        self.assertIsInstance(results, tuple, "Evaluation should return a tuple")
        self.assertEqual(len(results), 6, "Evaluation should return six metrics")
        print("Evaluation Results:", results)

if __name__ == '__main__':
    unittest.main()
