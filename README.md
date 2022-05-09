# AllenNLP-NER-ja: AllenNLP による日本語を対象とした固有表現抽出

## 実行環境の準備

```shell
pip install -U pip wheel setuptools poetry
```

```shell
poetry install
poetry run poe force-cuda11 # CUDA 11 を使いたい場合
```

## データセットの準備

- KWDLC (京都大学ウェブ文書リードコーパス)

```shell
git clone https://github.com/ku-nlp/KWDLC.git datasets/kwdlc/repo
allennlp make-dataset kwdlc --output-dir datasets/kwdlc --source datasets/kwdlc/repo/
```

- Stockmark NER Wikipedia データセット

```shell
git clone https://github.com/stockmarkteam/ner-wikipedia-dataset.git datasets/stockmark-ner-wiki/repo
allennlp make-dataset stockmark_ner_wiki --output-dir datasets/stockmark-ner-wiki --source datasets/stockmark-ner-wiki/repo/ner.json
```

## モデルの学習

- KWDLC (京都大学ウェブ文書リードコーパス)

```shell
CUDA_VISIBLE_DEVICES=0 GPU=0 allennlp train configs/kwdlc/bert.jsonnet -s outputs/kwdlc/bert
```

- Stockmark NER Wikipedia データセット

```shell
CUDA_VISIBLE_DEVICES=0 GPU=0 allennlp train configs/stockmark-ner-wiki/bert.jsonnet -s outputs/stockmark-ner-wiki/bert
```

## モデルの予測

- KWDLC (京都大学ウェブ文書リードコーパス)

```shell
CUDA_VISIBLE_DEVICES=0 allennlp predict \
  outputs/kwdlc/bert/model.tar.gz \
  datasets/kwdlc/tst_ner.txt \
  --output-file outputs/kwdlc/bert/tst_ner.jsonl \
  --cuda-device 0 --predictor sentence_tagger --use-dataset-reader
```

- Stockmark NER Wikipedia データセット

```shell
CUDA_VISIBLE_DEVICES=0 allennlp predict \
  outputs/stockmark-ner-wiki/bert/model.tar.gz \
  datasets/stockmark-ner-wiki/tst_ner.txt \
  --output-file outputs/kwdlc/stockmark-ner-wiki/tst_ner.jsonl \
  --cuda-device 0 --predictor sentence_tagger --use-dataset-reader
```

## Acknowledgements

- [AllenNLPによる自然言語処理 (3): BERTによる固有表現認識](https://colab.research.google.com/drive/13ga1yYYZkosGZy9ZinAB76blb-8k6yby?usp=sharing)
  - 上記ノートブックは[実践！AllenNLPによるディープラーニングを用いた自然言語処理 - Speaker Deck](https://speakerdeck.com/ikuyamada/shi-jian-allennlpniyorudeipuraninguwoyong-itazi-ran-yan-yu-chu-li )
- ku-nlp/KWDLC: Kyoto University Web Document Leads Corpus https://github.com/ku-nlp/KWDLC
- stockmarkteam/ner-wikipedia-dataset: Wikipediaを用いた日本語の固有表現抽出データセット https://github.com/stockmarkteam/ner-wikipedia-dataset 
