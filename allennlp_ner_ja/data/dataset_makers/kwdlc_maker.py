import glob
import logging
import os
import pathlib
import random
import re
from typing import List, Union

from allennlp_ner_ja.data.dataset_makers import DatasetMaker
from pyknp import BList

logger = logging.getLogger(__file__)


@DatasetMaker.register("kwdlc")
class KwdlcDatasetMaker(DatasetMaker):
    def __init__(
        self,
        output_dir: pathlib.Path,
        tng_file: str,
        val_file: str,
        tst_file: str,
        seed: int = 19950815,
    ) -> None:
        super().__init__(output_dir, tng_file, val_file, tst_file, seed)

    def get_ne_mrph_ids(
        self, last_mrph_id: int, ne_target: str, result: BList
    ) -> List[int]:
        """固有表現を構成する形態素id列を得る"""
        ne_mrph_ids: List[int] = []
        midasi = ""
        for i in range(last_mrph_id, -1, -1):
            ne_mrph_ids.insert(0, i)
            midasi = result.mrph_list()[i].midasi + midasi

            # 固有表現先頭まで来たので形態素id列を返す
            if midasi == ne_target:
                return ne_mrph_ids

        raise ValueError(f"Invalid result")

    def add_ne_tag_to_mrphs(self, result: BList) -> None:
        """基本句(「+」から始まる行)に付与されている固有表現タグを各形態素に付与"""

        for tag in result.tag_list():
            # 例: <NE:LOCATION:千里中央駅>
            match = re.search(r"<NE:(.+?):(.+?)>", tag.fstring)
            if not match:
                continue

            ne_type, ne_target = match.groups()

            # 曖昧性が高いなどの理由によりタグ付けが困難なものにはOPTIONALタグが付与されており、
            # このタグは対象としない。
            # (IREXの評価では、OPTIONALが付与されているものに対してシステムが何らかのタグを推定した場合、
            # それを誤りとみなさない処置が行われているが、ここでは簡易的なものとしてそのような評価は行わず、
            # 単にOPTIONALタグを無視したデータ作成とする。
            # IREXの評価の詳細は https://nlp.cs.nyu.edu/irex/NE/df990214.txt の「1.1 オプショナル」節を参照のこと)
            if ne_type == "OPTIONAL":
                continue

            # 固有表現の末尾の形態素が含まれる基本句に固有表現タグが付与されているので
            # その基本句内で固有表現末尾の形態素を探す
            for mrph in reversed(tag.mrph_list()):
                if not ne_target.endswith(mrph.midasi):
                    continue

                # 固有表現を構成する形態素id列を得る
                ne_mrph_ids = self.get_ne_mrph_ids(mrph.mrph_id, ne_target, result)
                # 各形態素にNEタグを付与
                for i, ne_mrph_id in enumerate(ne_mrph_ids):
                    target_mrph = result.mrph_list()[ne_mrph_id]
                    # 固有表現の先頭はラベルB、それ以外はラベルI
                    target_mrph.fstring += "<NE:{}:{}/>".format(
                        ne_type, "B" if i == 0 else "I"
                    )

    def write_file(self, out_file: pathlib.Path, docs: List[BList]) -> None:
        """データセットをファイルに書き出す"""
        logger.info(f"Write file to {out_file}")

        with open(out_file, "w") as f:
            for doc in docs:
                for result in doc:
                    for mrph in result.mrph_list():
                        match = re.search(r"<NE:(.+?):([BI])/>", mrph.fstring)
                        if match:
                            # B-PERSONのような形式の固有表現ラベルを作成
                            ne_tag = "{}-{}".format(match.group(2), match.group(1))
                        else:
                            # 固有表現ラベルの無い場合は"O"ラベルを付与
                            ne_tag = "O"
                        # 1カラム目に単語、4カラム目に固有表現ラベルを書く
                        # それ以外のカラムは利用しない
                        f.write("{} N/A N/A {}\n".format(mrph.midasi, ne_tag))
                    f.write("\n")

    def __call__(self, file_path: Union[str, pathlib.Path]) -> None:
        docs = []
        doc_files = sorted(
            glob.glob(os.path.join(file_path, "**/*.knp"), recursive=True)
        )

        for doc_file in doc_files:
            results = []
            with open(doc_file, "r") as rf:
                # 文書に含まれる文とその固有表現ラベルを読み込む
                buf = ""
                for line in rf:
                    buf += line
                    if "EOS" in line:
                        result = BList(buf)
                        self.add_ne_tag_to_mrphs(result)
                        results.append(result)
                        buf = ""
            docs.append(results)

        # データセットをランダムに並べ替える
        random.shuffle(docs)

        # データセットの80%を訓練データ、10%を検証データ、10%をテストデータとして用いる
        num_train = int(0.8 * len(docs))
        num_test = int(0.1 * len(docs))
        tng_docs = docs[:num_train]
        val_docs = docs[num_train:-num_test]
        tst_docs = docs[-num_test:]

        # データセットをファイルに書き込む
        self.write_file(self.tng_file_path, tng_docs)
        self.write_file(self.val_file_path, val_docs)
        self.write_file(self.tst_file_path, tst_docs)
