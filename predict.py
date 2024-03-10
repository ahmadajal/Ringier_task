import argparse
import json
import os
from typing import List

from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, Trainer

from train import load_datasets


def predict(predict_payload: str, model_name: str) -> List:
    """This function computes the class probabilities for the test data.

    Args:
        predict_payload: Path to the predict_payload.json file.
        model_name: Name of the model from huggingface model hub.

    Returns:
        A list of lists of class probabilities.
    """
    # Load the tokenized test data.
    pred_dataset = load_datasets(path=predict_payload, test=True)
    # Load the trained model
    ckpt_dirs = os.listdir(f"models/{model_name.split('/')[-1]}")
    # As we loaded the best model in the end of training, the last checkpoint
    # is the best one.
    last_ckpt = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    model = AutoModelForSequenceClassification.from_pretrained(
        f"models/{model_name.split('/')[-1]}/{last_ckpt}/"
    )
    trainer = Trainer(model)
    output = trainer.predict(pred_dataset)
    pred_probs = softmax(output.predictions, axis=1)
    return pred_probs.tolist()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_to_payload", type=str, help="path to predict_payload.json")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model name from the huggingface model hub.",
        default="google-bert/bert-base-cased",
    )
    args = argparser.parse_args()
    pred_probs = predict(args.path_to_payload, args.model_name)
    # Save the predictions.
    os.makedirs("predictions/", exist_ok=True)
    with open("predictions/probas.json", "w") as f:
        json.dump(pred_probs, f)
    print("predictions are save to predictions/probas.json")
