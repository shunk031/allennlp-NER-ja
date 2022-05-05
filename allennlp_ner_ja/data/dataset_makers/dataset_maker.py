import pathlib
from typing import Union

from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.common.util import prepare_environment


class DatasetMaker(Registrable):
    def __init__(
        self,
        output_dir: pathlib.Path,
        tng_file: str,
        val_file: str,
        tst_file: str,
        seed: int = 19950815,
    ) -> None:
        super().__init__()

        self._output_dir = output_dir
        self._tng_file = tng_file
        self._val_file = val_file
        self._tst_file = tst_file

        self._set_seed(seed)

    @property
    def tng_file_path(self) -> pathlib.Path:
        return self._output_dir / self._tng_file

    @property
    def val_file_path(self) -> pathlib.Path:
        return self._output_dir / self._val_file

    @property
    def tst_file_path(self) -> pathlib.Path:
        return self._output_dir / self._tst_file

    def _set_seed(self, seed: int) -> None:
        prepare_environment(
            Params(
                {
                    "random_seed": seed,
                    "numpy_seed": seed,
                    "pytorch_seed": seed,
                }
            )
        )

    def __call__(self, file_path: Union[str, pathlib.Path]) -> None:
        raise NotImplementedError
