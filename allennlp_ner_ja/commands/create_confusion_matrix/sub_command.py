import argparse

from allennlp.commands.predict import Predict
from allennlp.commands.subcommand import Subcommand

from .function import create_confusion_matrix


@Subcommand.register("create-confusion-matrix")
class CreateConfusionMatrix(Predict):
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        subparser = super().add_subparser(parser)
        subparser.add_argument("--test-file", type=str)
        subparser.set_defaults(func=create_confusion_matrix)
        return subparser
