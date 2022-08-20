import logging
from rich.progress import track

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import argparse
import requests
import time

from requests.exceptions import ConnectionError
from seqeval.metrics import accuracy_score, f1_score
from typing import Tuple, List, Any, Dict


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:

    tokens_s = []
    labels_s = []

    tokens = []
    labels = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if line.startswith("#\t"):
                tokens = []
                labels = []
            elif line == "":
                tokens_s.append(tokens)
                labels_s.append(labels)
            else:
                token, label = line.split("\t")
                tokens.append(token)
                labels.append(label)

    assert len(tokens_s) == len(labels_s)

    return tokens_s, labels_s


def main(test_path: str, endpoint: str, batch_size=32):

    try:
        tokens_s, labels_s = read_dataset(test_path)
    except FileNotFoundError as e:
        logging.error(f"Evaluation crashed because {test_path} does not exist")
        exit(1)
    except Exception as e:
        logging.error(
            f"Evaluation crashed. Most likely, the file you gave is not in the correct format"
        )
        logging.error(f"Printing error found")
        logging.error(e, exc_info=True)
        exit(1)

    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(
                f"Impossible to establish a connection to the server even after 10 tries"
            )
            logging.error(
                "The server is not booting and, most likely, you have some error in build_model or StudentClass"
            )
            logging.error(
                "You can find more information inside logs/. Checkout both server.stdout and, most importantly, server.stderr"
            )
            exit(1)

        logging.info(f"Waiting 10 second for server to go up: trial {i}/{max_try}")
        time.sleep(10)

        try:
            response = requests.post(
                endpoint, json={"tokens_s": [["My", "name", "is", "Robin", "Hood"]]}
            ).json()
            response["predictions_s"]
            logging.info("Connection succeded")
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    predictions_s = []

    for i in track(range(0, len(tokens_s), batch_size), description="Evaluating"):
        batch = tokens_s[i : i + batch_size]
        try:
            response = requests.post(endpoint, json={"tokens_s": batch}).json()
            predictions_s += response["predictions_s"]
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    flat_labels_s = flat_list(labels_s)
    flat_predictions_s = flat_list(predictions_s)

    label_distribution = count(flat_labels_s)
    pred_distribution = count(flat_predictions_s)

    print(f"# instances: {len(flat_list(labels_s))}")

    keys = set(label_distribution.keys()) | set(pred_distribution.keys())
    for k in keys:
        print(
            f"\t# {k}: ({label_distribution.get(k, 0)}, {pred_distribution.get(k, 0)})"
        )

    acc = accuracy_score(labels_s, predictions_s)
    f = f1_score(labels_s, predictions_s, average="macro")

    print(f"# accuracy: {acc:.4f}")
    print(f"# f1: {f:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    args = parser.parse_args()

    main(test_path=args.file, endpoint="http://127.0.0.1:12345")
