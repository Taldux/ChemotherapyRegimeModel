"""
@file main.py
@brief CLI entry point for the chemotherapy regime model pipeline.
"""
import argparse
from src.train import ALL_MODEL_KEYS, DATA_SOURCE_DIRS, main as train_main


def main():
    """
    @brief Parse CLI arguments and launch the training / evaluation pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='semi', choices=list(DATA_SOURCE_DIRS),
                        help='Which dataset contract to use (default: semi)')
    parser.add_argument('--models', nargs='+', default=None,
                        choices=ALL_MODEL_KEYS,
                        help='Which models to train (default: all)')
    args = parser.parse_args()

    if args.models is None:
        train_main(data_source=args.data)
    else:
        train_main(active_keys=args.models, data_source=args.data)


if __name__ == "__main__":
    main()
