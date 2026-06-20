from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import os
import sys
# api.py'ın bulunduğu klasörün bir üst klasörünü (kök dizini) sys.path'e ekler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.backends.model1_model2 import Model1Model2Backend
from inference.backends.global_cnn import GlobalCNNBackend
from inference.backends.global_cnn_svm import GlobalCNNSVMBackend
from inference.backends.transfer_learning import TransferLearning08Backend

app = Flask(__name__)

# -----------------------------
# Backend Registry
# -----------------------------
BACKENDS = {
    "model1_model2": Model1Model2Backend(device="cpu"),
    "global_cnn": GlobalCNNBackend(device="cpu"),
    "cnn_svm": GlobalCNNSVMBackend(device="cpu"),
    "transfer_learning": TransferLearning08Backend(device="cpu"),
}

DEFAULT_BACKEND = "cnn_svm"


# -----------------------------
# Backend method adapter
# -----------------------------
def _run_predict(backend_obj, img_path: str):
    """
    Backend'ler arasında predict method isim farkını handle eder.
    """
    if hasattr(backend_obj, "predict_one"):
        return backend_obj.predict_one(img_path)

    if hasattr(backend_obj, "predict_path"):
        return backend_obj.predict_path(img_path)

    raise AttributeError("Backend must implement predict_one() or predict_path().")


# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "available_backends": list(BACKENDS.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():

    backend_name = request.form.get("backend", DEFAULT_BACKEND)

    if backend_name not in BACKENDS:
        return jsonify({
            "error": f"Unknown backend '{backend_name}'",
            "available_backends": list(BACKENDS.keys())
        }), 400

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (form key: image)"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, filename)
        file.save(img_path)

        try:
            result = _run_predict(BACKENDS[backend_name], img_path)

            # Backend adı her zaman response içinde olsun
            if isinstance(result, dict):
                result["backend"] = backend_name
            else:
                result = {
                    "backend": backend_name,
                    "result": result
                }

        except Exception as e:
            return jsonify({
                "error": str(e),
                "backend": backend_name
            }), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)