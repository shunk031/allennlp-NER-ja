import argparse
import json
import os
from itertools import chain

from allennlp.commands.predict import _predict
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(args: argparse.Namespace) -> None:

    if not os.path.exists(args.test_file):
        _predict(args)

    predictions = []

    with open(args.output_file, "r") as rf:
        for line in rf:
            predictions.append(json.loads(line))

    gold_labels = []
    with open(args.test_file, "r") as rf:
        for line in rf:

            if line == "\n":
                continue

            gold_labels.append(line.split()[-1])

    pred_tags = list(chain.from_iterable([p["tags"] for p in predictions]))
    assert len(gold_labels) == len(pred_tags)
    print(confusion_matrix(y_true=gold_labels, y_pred=pred_tags))
