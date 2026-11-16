"""Side-by-side comparison: base Qwen3-1.7B vs. the DPO-humanized version."""
import os

from flask import Flask, jsonify, render_template, request
from mlx_lm import generate, load

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
# Fused model produced by dpo_minimal.ipynb (or a Hugging Face repo id)
DPO_MODEL = os.getenv("DPO_MODEL", "./tests_dpo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

app = Flask(__name__)

print(f"Loading base model: {BASE_MODEL}")
base_model, base_tokenizer = load(BASE_MODEL)
print(f"Loading DPO model: {DPO_MODEL}")
dpo_model, dpo_tokenizer = load(DPO_MODEL)


def split_think(text):
    """Separate Qwen3 <think>...</think> reasoning from the final answer."""
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        return think.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


def respond(model, tokenizer, message):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    think, answer = split_think(generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS))
    return {"think": think, "answer": answer}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    return jsonify({
        "pre": respond(base_model, base_tokenizer, message),
        "post": respond(dpo_model, dpo_tokenizer, message),
    })


if __name__ == "__main__":
    app.run(port=5001)
