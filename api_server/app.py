import json

from flask import Flask, jsonify, request
from model_utils import load_model_for_inference

from src.data.utils import clean_text
from src.metadata import metadata

app = Flask(__name__)

model = load_model_for_inference(
    metadata.SAVED_MODELS_DIR / "latest-bf16.ckpt",
    metadata.SAVED_MODELS_DIR / "idx_to_label.json",
)
model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify(
                {
                    "statusCode": 400,  # Bad Request
                    "headers": {
                        "Content-Type": "application/json",
                    },
                    "body": "Empty string provided!",
                }
            )
        text = [clean_text(text)]
        labels = model.get_predictions(text)
        return jsonify(
            {
                "statusCode": 200,  # OK
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"sentiment": labels}),
            }
        )
    except Exception as e:
        print(repr(e))
        return jsonify(
            {
                "statusCode": 500,  # Internal Server Error;
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"error": repr(e)}),
            }
        )


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
