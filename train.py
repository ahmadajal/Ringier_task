import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from preprocess import preprocess_data

logging.basicConfig(filename="train_log.log", encoding="utf-8")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TAXONOMY_PATH = "data/taxonomy_mappings.json"
# Set the seed for reproducibility.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Training:
    def __init__(
        self, path_to_train: str, lr: float = 5e-5, batch_size: int = 16, epochs: int = 3
    ) -> None:
        """The main training class.

        Args:
            path_to_train: Path to the training data in json format.
            lr: Learning rate. Defaults to 5e-5.
            batch_size: Training batch size. Defaults to 16.
            epochs: Number of epochs for training. Defaults to 3.
        """
        self.path = path_to_train
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

    def load_datasets(self, split: bool = True) -> List[Dataset]:
        """
        This method preprocess the raw data and convert it to Huggingface dataset. Then
        it tokenize the dataset with the proper tokenizer and converts it to torch format.
        Finally, it splits the data to training and test, maintining the balance of the
        class labels in both datasets.

        Args:
            split: If True the dataset will be splitted to training and test.

        Returns:
            List: train and test datasets.
        """
        data = preprocess_data(
            path_to_data=self.path, test=False, path_to_taxonomy_mappings=TAXONOMY_PATH
        )
        logging.info("Raw data preprocessing finished!")

        # Create Huggingface dataset
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        # Tokenize the dataset for the model.
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x["fullText"], padding="max_length", truncation=True),
            batched=True,
        )
        # We have to remove the string type columns!
        tokenized_dataset = tokenized_dataset.remove_columns(["fullText"])
        tokenized_dataset = tokenized_dataset.remove_columns(["title"])
        # Convert the dataset to torch format
        tokenized_dataset.set_format("torch")
        if split:
            # Split the dataset to 20% test and 80% train.
            train_inds, test_inds = train_test_split(
                range(len(tokenized_dataset)),
                test_size=0.2,
                stratify=tokenized_dataset["labels"].argmax(dim=1).numpy(),
                random_state=SEED,
            )
            train_dataset = tokenized_dataset.select(train_inds)
            test_dataset = tokenized_dataset.select(test_inds)
            return train_dataset, test_dataset
        else:
            return tokenized_dataset

    def train(self):
        """This method loads the train and test datasets and trains the model."""
        X_train, X_test = self.load_datasets()
        logging.info("Training and Test data loaded.")
        num_labels = len(X_train[0]["labels"])
        train_dataloader = DataLoader(X_train, shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(X_test, batch_size=self.batch_size)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-cased", num_labels=num_labels
        )
        # Model optimizer and learning rate scheduler.
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.epochs * len(train_dataloader),
        )
        # Put the model in the appropriate device.
        self.model.to(device)
        # Start the training loop.
        logging.info("Training starts ...")
        progress_bar = tqdm(range(self.epochs * len(train_dataloader)))
        for epoch in range(self.epochs):
            self.model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            acc_epoch = self.evaluate(test_dataloader)
            logging.info(f"Validation accuracy after epoch {epoch}: {acc_epoch}")
        logging.info("Training finished!")
        acc = self.evaluate(test_dataloader)
        logging.info(f"Test accuracy: {acc}")
        print(f"Test accuracy: {acc}")

    def evaluate(self, test_dataloader):
        correct_preds = np.array([])
        self.model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            true_labels = batch["labels"].argmax(dim=-1)
            correct_preds = np.concatenate(
                (correct_preds, (preds == true_labels).cpu().numpy())
            )
        return sum(correct_preds) / len(correct_preds)


if __name__ == "__main__":
    training = Training(path_to_train="data/train_data.json")
    training.train()
    os.makedirs("models/", exist_ok=True)
    training.model.save_pretrained("models/bert-based-trained")
