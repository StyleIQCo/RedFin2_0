"""Run the whole offline pipeline: generate → train AVM → train recommender."""
from data.generate import generate
from src.avm.train import main as train_avm
from src.recommender.train import main as train_rec


def main() -> None:
    generate()
    train_avm()
    train_rec()


if __name__ == "__main__":
    main()
