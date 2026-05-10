import base64
import os
from typing import Any, Dict

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from asl_predictor import ASLPredictor


def _data_url_to_bgr(data_url: str) -> np.ndarray:
    # Expect: "data:image/jpeg;base64,...."
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    b64 = data_url.split(",", 1)[1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


app = Flask(__name__)
predictor = ASLPredictor(model_path=os.environ.get("ASL_MODEL_PATH", "cnn8grps_rad1_model.h5"))


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    body: Dict[str, Any] = request.get_json(force=True, silent=False)
    data_url = body.get("imageDataUrl")
    if not data_url:
        return jsonify({"ok": False, "error": "Missing imageDataUrl"}), 400

    try:
        frame_bgr = _data_url_to_bgr(data_url)
        result = predictor.predict_from_bgr(frame_bgr)
        return jsonify(
            {
                "ok": True,
                "symbol": result.symbol,
                "hasHand": bool(result.skeleton_bgr is not None),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Local dev server
    app.run(host="127.0.0.1", port=5000, debug=True)

