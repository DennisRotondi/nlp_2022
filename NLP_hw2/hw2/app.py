from flask import Flask, request, jsonify

from stud.implementation import build_model_34, build_model_234, build_model_1234

app = Flask(__name__)

mandatory_languages = ["EN"]
extras_languages = ["FR", "ES"]

all_languages = mandatory_languages + extras_languages
models_34, models_234, models_1234 = dict(), dict(), dict()

for lang in mandatory_languages:
    models_34[lang] = build_model_34(device="cpu", language=lang)

for lang in extras_languages:
    try:
        models_34[lang] = build_model_34(device="cpu", language=lang)
    except:
        pass

for lang in all_languages:
    try:
        models_234[lang] = build_model_234(device="cpu", language=lang)
    except:
        pass

for lang in all_languages:
    try:
        models_1234[lang] = build_model_1234(device="cpu", language=lang)
    except:
        pass


def prepare_data(data):
    data_34 = data
    data_234 = {
        "words": data["words"],
        "lemmas": data["lemmas"],
        "predicates": [1 if p != "_" else 0 for p in data["predicates"]],
    }
    data_1234 = {
        "words": data["words"],
        "lemmas": data["lemmas"],
    }

    return data_34, data_234, data_1234


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):
    try:
        json_body = request.json
        data = json_body["data"]
        language = json_body["language"]

        if language not in all_languages:
            return (
                {
                    "error": "Bad request",
                    "message": f"Language `{language}` not supported, please choose one between `EN`, `FR` and `ES`",
                },
                400,
            )

        data_34, data_234, data_1234 = prepare_data(data)

        predictions_34, predictions_234, predictions_1234 = None, None, None

        if language in models_34:
            predictions_34 = models_34[language].predict(data_34)
        if language in models_234:
            predictions_234 = models_234[language].predict(data_234)
        if language in models_1234:
            predictions_1234 = models_1234[language].predict(data_1234)
    except Exception as e:

        app.logger.error(e, exc_info=True)
        return (
            {
                "error": "Bad request",
                "message": "There was an error processing the request. Please check logs/server.stderr",
            },
            400,
        )

    return jsonify(
        data=data,
        predictions_34=predictions_34,
        predictions_234=predictions_234,
        predictions_1234=predictions_1234,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345)
