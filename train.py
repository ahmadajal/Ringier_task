import argparse
import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from preprocess import preprocess_data

TAXONOMY_PATH = "data/taxonomy_mappings.json"
# Set the seed for reproducibility.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_datasets(path: str, model_name: str, test: bool = False) -> Dataset:
    """
    This function preprocesses the raw data and converts it to Huggingface dataset. Then
    it tokenize the dataset with the proper tokenizer and converts it to torch format.
    Finally, it splits the data to training and test, maintining the balance of the
    class labels in both datasets.

    Args:
        path: Path to the training dataset json file.
        model_name: Name of the model from huggingface model hub.
        test: If True it indicates that we are loading a test dataset for prediction
        and hence we do not need to do training/test splitting. Defaults to False.

    Returns:
        List: train and test datasets.
    """
    # For processing the prediction data, we need to set the test flag to True, since
    # there are no labels in the data. We can use the split flag to do this!
    data = preprocess_data(
        path_to_data=path, test=test, path_to_taxonomy_mappings=TAXONOMY_PATH
    )
    logging.info("Raw data preprocessing finished!")

    # Create Huggingface dataset
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    # Tokenize the dataset for the model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["fullText"], padding="max_length", truncation=True),
        batched=True,
    )
    # We have to remove the string type columns!
    tokenized_dataset = tokenized_dataset.remove_columns(["fullText"])
    tokenized_dataset = tokenized_dataset.remove_columns(["title"])
    # Convert the dataset to torch format
    tokenized_dataset.set_format("torch")
    if not test:
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


def train(path: str, model_name: str, batch_size: int) -> None:
    """This method loads the train and test datasets and trains the model.

    Args:
        path: Path to the training dataset json file.
        model_name: Name of the model from huggingface model hub.
        batch_size: Batch size for training and evaluation.
    """
    X_train, X_test = load_datasets(path, model_name)
    logging.info("Training and Test data loaded.")
    num_labels = len(X_train[0]["labels"])
    # Since we have multiple labels per training data, we set the problem type
    # to mulit-label. In this case, binary cross entropy loss will be used to
    # compute the loss for each label.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    )
    training_args = TrainingArguments(
        f"models/{model_name.split('/')[-1]}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=5,
        num_train_epochs=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_test,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)
        ],
    )
    # Start training ...
    logging.info("Training starts ...")
    trainer.train()
    eval_metrics = trainer.evaluate()
    logging.info(
        f"Metrics on the validation data. \n \
                 accuracy: {eval_metrics['eval_accuracy']} \n \
                 precision: {eval_metrics['eval_precision']} \n \
                 recall: {eval_metrics['eval_recall']}"
    )
    logging.info("Training finished!")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply Sigmoid to get the confidence for each class label.
    preds = expit(logits)
    num_classes = labels.shape[-1]
    # Use the labels with confidence>0.5 as correct labels.
    labels = [
        np.arange(labels.shape[-1])[labels[i] > 0.5].tolist() for i in range(labels.shape[0])
    ]
    # Same for the predictions
    preds = [
        np.arange(preds.shape[-1])[preds[i] > 0.5].tolist() for i in range(preds.shape[0])
    ]
    # Convert labels and predictions to multi-label format.
    mlb = MultiLabelBinarizer(classes=range(num_classes))
    labels = mlb.fit_transform(labels)
    preds = mlb.fit_transform(preds)

    # Calculate the metrics.
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model name from the huggingface model hub.",
        default="google-bert/bert-base-cased",
    )
    argparser.add_argument("--batch_size", type=int, default=8)
    args = argparser.parse_args()
    logging.basicConfig(
        filename="train_log.log", filemode="w", encoding="utf-8", level=logging.INFO
    )
    train(path="data/train_data.json", model_name=args.model_name, batch_size=args.batch_size)
