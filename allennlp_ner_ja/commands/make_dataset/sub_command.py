import argparse
import pathlib

from allennlp.commands.subcommand import Subcommand
from allennlp_ner_ja.data.dataset_makers import DatasetMaker

from .function import make_dataset_from_args


@Subcommand.register("make-dataset")
class MakeDataset(Subcommand):
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        subparser = parser.add_parser(
            self.name, description="Command for making dataset"
        )
        subparser.add_argument(
            "dataset_type",
            type=str,
            default="kwdlc",
            choices=DatasetMaker.list_available(),
        )
        subparser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            default=pathlib.Path(__file__).resolve().parents[3] / "datasets",
        )
        subparser.add_argument(
            "--source",
            type=pathlib.Path,
            required=True,
            help="ファイルまたはディレクトリのパスを指定する",
        )
        subparser.add_argument("--tng-file", type=str, default="kwdlc_ner_tng.txt")
        subparser.add_argument("--val-file", type=str, default="kwdlc_ner_val.txt")
        subparser.add_argument("--tst-file", type=str, default="kwdlc_ner_tst.txt")
        subparser.set_defaults(
            func=lambda args: make_dataset_from_args(
                dataset_type=args.dataset_type,
                source=args.source,
                output_dir=args.output_dir,
                tng_file=args.tng_file,
                val_file=args.val_file,
                tst_file=args.tst_file,
            )
        )
        return subparser
