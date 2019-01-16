import argparse
import os
import sys

from botshot_nlu.config import TrainingHelper, read_training_config, ParseHelper


def start_training(args):

    config = read_training_config(args.config)
    config_dir = os.path.dirname(args.config)
    model_path = args.model_path
    crossvalidate = args.crossvalidate
    training = TrainingHelper(config=config, entities=args.entities, config_dir=config_dir, save_path=model_path, crossvalidate=crossvalidate)
    training.start()


def start_parse(args):
    config = read_training_config(args.config)
    helper = ParseHelper.load(args.model_path)
    print("Enter query: (exit with Ctrl+D)")
    for line in sys.stdin:
        helper.parse(line)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("action", choices=['train', 'parse'])
    argp.add_argument("--config", type=str, default='nlu-config.yml')
    #argp.add_argument("--examples", nargs='+', type=str)
    argp.add_argument("--entities", nargs='+', type=str)
    argp.add_argument("--model-path", type=str)
    argp.add_argument("--crossvalidate", action='store_true')
    args = argp.parse_args()
    if args.action == 'train':
        start_training(args)
    elif args.action == 'parse':
        start_parse(args)


if __name__ == "__main__":
    main()