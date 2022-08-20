import logging

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import argparse
import json
import pprint
import requests
import time

from requests.exceptions import ConnectionError
from typing import Tuple, List, Any, Dict
from rich.progress import track

import utils


def main(test_path: str, endpoint: str, language: str):
    try:
        sentences, labels = utils.read_dataset(test_path)
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
                endpoint,
                json={
                    "data": sentences[list(sentences.keys())[0]],
                    "language": language,
                },
            ).json()
            response["predictions_34"]
            logging.info("Connection succeded")
            break
        except ConnectionError as e:
            print(e)
            continue
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            # logging.error(f"Response was: {response}")
            if response["error"] == "Bad request":
                logging.error(f"Error message: {response['message']}", exc_info=True)
            else:
                logging.error(e, exc_info=True)
            exit(1)

    predictions_34 = {}
    predictions_234 = {}
    predictions_1234 = {}

    for sentence_id in track(sentences, description="Evaluating"):
        sentence = sentences[sentence_id]
        try:
            response = requests.post(
                endpoint, json={"data": sentence, "language": language}
            ).json()
            predictions_34[sentence_id] = response["predictions_34"]
            predictions_34[sentence_id]["roles"] = {
                int(i): p for i, p in predictions_34[sentence_id]["roles"].items()
            }
            if response["predictions_234"]:
                predictions_234[sentence_id] = response["predictions_234"]
                predictions_234[sentence_id]["roles"] = {
                    int(i): p for i, p in predictions_234[sentence_id]["roles"].items()
                }
            if response["predictions_1234"]:
                predictions_1234[sentence_id] = response["predictions_1234"]
                predictions_1234[sentence_id]["roles"] = {
                    int(i): p for i, p in predictions_1234[sentence_id]["roles"].items()
                }
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            if response["error"] == "Bad request":
                logging.error(f"Error message: {response['message']}", exc_info=True)
            else:
                logging.error(f"Response was: {response}")
                logging.error(e, exc_info=True)
            exit(1)

    print("MODEL: ARGUMENT IDENTIFICATION + ARGUMENT CLASSIFICATION")
    argument_identification_results = utils.evaluate_argument_identification(
        labels, predictions_34
    )
    argument_classification_results = utils.evaluate_argument_classification(
        labels, predictions_34
    )
    print(utils.print_table("argument identification", argument_identification_results))
    print(utils.print_table("argument classification", argument_classification_results))

    if predictions_234:
        print(
            "MODEL: PREDICATE DISAMBIGUATION + ARGUMENT IDENTIFICATION + ARGUMENT CLASSIFICATION"
        )
        predicate_disambiguation_results = utils.evaluate_predicate_disambiguation(
            labels, predictions_234
        )
        print(
            utils.print_table(
                "predicate disambiguation", predicate_disambiguation_results
            )
        )

        argument_identification_results = utils.evaluate_argument_identification(
            labels, predictions_234
        )
        argument_classification_results = utils.evaluate_argument_classification(
            labels, predictions_234
        )
        print(
            utils.print_table(
                "argument identification", argument_identification_results
            )
        )
        print(
            utils.print_table(
                "argument classification", argument_classification_results
            )
        )

    if predictions_1234:
        print(
            "MODEL: PREDICATE IDENTIFICATION + PREDICATE DISAMBIGUATION + ARGUMENT IDENTIFICATION + ARGUMENT CLASSIFICATION"
        )
        predicate_identification_results = utils.evaluate_predicate_identification(
            labels, predictions_1234
        )
        print(
            utils.print_table(
                "predicate identification", predicate_identification_results
            )
        )
        predicate_disambiguation_results = utils.evaluate_predicate_disambiguation(
            labels, predictions_1234
        )
        print(
            utils.print_table(
                "predicate disambiguation", predicate_disambiguation_results
            )
        )

        argument_identification_results = utils.evaluate_argument_identification(
            labels, predictions_1234
        )
        argument_classification_results = utils.evaluate_argument_classification(
            labels, predictions_1234
        )
        print(
            utils.print_table(
                "argument identification", argument_identification_results
            )
        )
        print(
            utils.print_table(
                "argument classification", argument_classification_results
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    parser.add_argument("language", type=str, help="Language of the dataset")
    args = parser.parse_args()

    main(test_path=args.file, endpoint="http://127.0.0.1:12345", language=args.language)
