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

## モデルの学習

- KWDLC (京都大学ウェブ文書リードコーパス)

```shell
CUDA_VISIBLE_DEVICES=1 GPU=0 allennlp train configs/kwdlc/bert.jsonnet -s outputs/kwdlc/bert
```

## Acknowledgements

- [AllenNLPによる自然言語処理 (3): BERTによる固有表現認識](https://colab.research.google.com/drive/13ga1yYYZkosGZy9ZinAB76blb-8k6yby?usp=sharing)
