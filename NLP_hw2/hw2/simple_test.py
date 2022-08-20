import argparse

import utils
import logging
from stud.implementation import build_model_34

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def main(test_path: str):
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
    model = build_model_34("cpu", "EN")
    predicted_sentences = model.predict(next(iter(sentences.values())))
    print(predicted_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    args = parser.parse_args()
    main(test_path=args.file)