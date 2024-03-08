import argparse
import json
import logging
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from train import Training

logging.basicConfig(filename="test_log.log", encoding="utf-8")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
argparser = argparse.ArgumentParser()
argparser.add_argument("--path_to_payload", type=str, help="path to predict_payload.json")
args = argparser.parse_args()


def predict(predict_payload: str) -> List:
    """This function computes the class probabilities for the test data.

    Args:
        predict_payload: Path to the predict_payload.json file.

    Returns:
        A list of lists of class probabilities.
    """
    # Load the tokenized test data.
    training = Training(path_to_train=predict_payload)
    pred_dataset = training.load_datasets(split=False)
    logging.info("Prediction data loaded.")
    # We need to load the data in batches, in case the prediction dataset is large!
    pred_dataloader = DataLoader(pred_dataset, batch_size=training.batch_size)
    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained("models/bert-based-trained/")
    model.eval()
    pred_probs = []
    for batch in pred_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        probabilities = torch.nn.Softmax(dim=1)(output.logits)
        pred_probs.extend(probabilities.cpu().numpy().tolist)
    return pred_probs


if __name__ == "__main__":
    pred_probs = predict(args.path_to_payload)
    # Save the predictions.
    os.makedirs("predictions/", exist_ok=True)
    with open("predictions/probas.json", "w") as f:
        json.dump(pred_probs, f)
