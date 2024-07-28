import gradio as gr
from model_utils import load_model_for_inference

from src.data.utils import clean_text
from src.metadata import metadata

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
    title="Sentiment analysis demo",
)


if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=5000)
