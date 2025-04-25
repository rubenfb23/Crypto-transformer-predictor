import argparse

from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.visualization.dashboard import Dashboard


def main():
    """Punto de entrada para la aplicación."""
    print("Crypto-transformer-predictor started.")
    parser = argparse.ArgumentParser(description="Crypto Transformer Predictor")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "dashboard"],
        help="Modo de operación",
        default=None,
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "evaluate":
        evaluate_model()
    elif args.mode == "dashboard":
        Dashboard().start()
    # no se especificó modo


if __name__ == "__main__":
    main()
