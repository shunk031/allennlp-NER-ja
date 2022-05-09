import json
import logging
import os
import pathlib
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from allennlp_ner_ja.data.dataset_makers import DatasetMaker

logger = logging.getLogger(__name__)


@dataclass
class NerEntity(object):
    name: str
    span: List[int]
    type: str


@dataclass
class NerData(object):
    curid: str
    text: str
    entities: List[NerEntity]

    def __post_init__(self):
        self.entities = [NerEntity(**entity) for entity in self.entities]


@DatasetMaker.register("stockmark_ner_wiki")
class StockmarkNerWikiDatasetMaker(DatasetMaker):
    def __init__(
        self,
        output_dir: pathlib.Path,
        tng_file: str,
        val_file: str,
        tst_file: str,
        mecab_dic: Optional[str] = None,
        mecab_option: Optional[str] = None,
        seed: int = 19950815,
    ) -> None:
        super().__init__(output_dir, tng_file, val_file, tst_file, seed)
        import fugashi

        mecab_option = mecab_option or ""
        if mecab_dic is not None:
            if mecab_dic == "unidic_lite":
                import unidic_lite  # NOQA

                dic_dir = unidic_lite.DICDIR
            elif mecab_dic == "unidic":
                import unidic  # NOQA

                dic_dir = unidic.DICDIR
            elif mecab_dic == "ipadic":
                import ipadic

                dic_dir = ipadic.DICDIR
            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = "-d {} -r {} ".format(dic_dir, mecabrc) + mecab_option

        self.tagger = fugashi.GenericTagger(mecab_option)

    def create_tmp_ne_tags(self, ner_data: NerData) -> List[str]:
        tmp_ne_tags = ["O" for _ in range(len(ner_data.text))]
        for entity in ner_data.entities:
            for i, s in enumerate(range(*entity.span)):
                if i == 0:
                    tmp_ne_tags[s] = f"B-{entity.type}"
                else:
                    tmp_ne_tags[s] = f"I-{entity.type}"
        return tmp_ne_tags

    def create_text_ne_tags(
        self, ner_data: NerData, tmp_ne_tags: List[str]
    ) -> List[Tuple[str, str]]:

        assert len(tmp_ne_tags) == len(ner_data.text)

        text_ne_tags = []

        curr = 0
        for word in self.tagger(ner_data.text):
            ne_tag, *_ = tmp_ne_tags[curr : curr + len(word.surface)]
            text_ne_tags.append((word.surface, ne_tag))
            curr += len(word.surface)

        return text_ne_tags

    def write_file(
        self, out_file: pathlib.Path, docs: List[List[Tuple[str, str]]]
    ) -> None:
        logger.info(f"Write file to {out_file}")

        with open(out_file, "w") as wf:
            for text_ne_tags in docs:
                for text, ne_tag in text_ne_tags:
                    wf.write(f"{text} N/A N/A {ne_tag}\n")
                wf.write("\n")

    def __call__(self, file_path: Union[str, pathlib.Path]) -> None:
        with open(file_path, "r") as rf:
            ner_json_list = json.load(rf)

        text_ne_tags_list: List[List[Tuple[str, str]]] = []
        for ner_json in ner_json_list:
            ner_data = NerData(**ner_json)

            ne_tags = self.create_tmp_ne_tags(ner_data)
            text_ne_tags = self.create_text_ne_tags(ner_data, tmp_ne_tags=ne_tags)
            text_ne_tags_list.append(text_ne_tags)

        random.shuffle(text_ne_tags_list)

        num_train = int(0.8 * len(text_ne_tags_list))
        num_test = int(0.1 * len(text_ne_tags_list))

        tng_docs = text_ne_tags_list[:num_train]
        val_docs = text_ne_tags_list[num_train:-num_test]
        tst_docs = text_ne_tags_list[-num_test:]

        self.write_file(self.tng_file_path, tng_docs)
        self.write_file(self.val_file_path, val_docs)
        self.write_file(self.tst_file_path, tst_docs)
