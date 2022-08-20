from typing import List

from stud.implementation import build_model


def main(sentences: List[List[str]]):

    model = build_model("cpu")
    predicted_sentences = model.predict(sentences)

    for sentence, tagged_sentence in zip(sentences, predicted_sentences):
        print(f"# sentence = {sentence}")
        for i, (token, tag) in enumerate(zip(sentence, tagged_sentence)):
            print(f"{i}\t{token}\t{tag}")
        print()


if __name__ == "__main__":
    main([["My", "name", "is", "Robin", "Hood"]])
