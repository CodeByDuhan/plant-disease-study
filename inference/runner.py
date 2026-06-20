import argparse
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.backends.model1_model2 import Model1Model2Backend
from inference.backends.global_cnn import GlobalCNNBackend
from inference.backends.global_cnn_svm import GlobalCNNSVMBackend
from inference.backends.transfer_learning import TransferLearning08Backend


def build_backend(name: str, device: str):
    name = name.lower()

    if name == "model1_model2":
        return Model1Model2Backend(device=device)

    elif name == "global_cnn":
        return GlobalCNNBackend(device=device)

    elif name == "global_cnn_svm":
        return GlobalCNNSVMBackend(device=device)

    elif name == "transfer_learning":
        return TransferLearning08Backend(device=device)  

    else:
        raise ValueError(f"Unknown backend: {name}")


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Inference Runner")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="global_cnn_svm",
        choices=["model1_model2", "global_cnn", "global_cnn_svm","transfer_learning"],
        help="Inference backend to use",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu / mps / cuda",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-K predictions (for global models)",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    backend = build_backend(args.backend, args.device)

    # model1_model2 predict_one, diğerleri predict_path kullanıyor olabilir
    if hasattr(backend, "predict_one"):
        result = backend.predict_one(str(image_path))
    else:
        result = backend.predict_path(str(image_path), topk=args.topk)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
