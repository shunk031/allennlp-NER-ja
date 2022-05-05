import pathlib

from allennlp.common.params import Params
from allennlp_ner_ja.data.dataset_makers import DatasetMaker


def make_dataset(
    dataset_type: str,
    source: pathlib.Path,
    output_dir: pathlib.Path,
    tng_file: str,
    val_file: str,
    tst_file: str,
) -> None:

    param_dict = {
        "type": dataset_type,
        "output_dir": output_dir,
        "tng_file": tng_file,
        "val_file": val_file,
        "tst_file": tst_file,
    }

    dataset_maker = DatasetMaker.from_params(Params(param_dict))
    dataset_maker(source)


def make_dataset_from_args(
    dataset_type: str,
    source: pathlib.Path,
    output_dir: pathlib.Path,
    tng_file: str,
    val_file: str,
    tst_file: str,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    make_dataset(
        dataset_type=dataset_type,
        source=source,
        output_dir=output_dir,
        tng_file=tng_file,
        val_file=val_file,
        tst_file=tst_file,
    )
