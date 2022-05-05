import glob
import os
import pathlib
from typing import Dict, Iterable, Optional

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from kyoto_reader import KyotoReader


@DatasetReader.register("kwdlc")
class KwdlcDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        kyoto_reader_n_jobs: int = 4,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            max_instances,
            manual_distributed_sharding,
            manual_multiprocess_sharding,
            serialization_dir,
        )
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self._kyoto_reader_n_jobs = kyoto_reader_n_jobs

    def _read(self, file_path: str) -> Iterable[Instance]:
        reader = KyotoReader(source=file_path, extract_nes=True)

        docs = reader.process_all_documents(n_jobs=self._kyoto_reader_n_jobs)
        for doc in docs:
            midasi_list = [morph.midasi for morph in doc.mrph_list()]

            breakpoint()
