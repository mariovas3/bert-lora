import gradio as gr
from flask import Flask
from model_utils import load_model_for_inference

from src.data.utils import clean_text
from src.metadata import metadata

app = Flask(__name__)

model = load_model_for_inference(
    metadata.SAVED_MODELS_DIR / "latest-bf16.ckpt",
    metadata.SAVED_MODELS_DIR / "idx_to_label.json",
)
model.eval()


def predict(text):
    if not text:
        return ["Empty string provided!"]
    text = [clean_text(text)]
    labels = model.get_predictions(text)
    return labels


interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
)


@app.route("/", methods=["GET"])
def gradio_ui():
    return interface.launch(share=True, inline=True)


def main():
    app.run(host="0.0.0.0", debug=False)


if __name__ == "__main__":
    main()
