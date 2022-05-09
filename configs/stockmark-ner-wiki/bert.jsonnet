local cuda_device = std.parseInt(std.extVar("GPU"));

local seed = 19950815;
local num_epochs = 30;
local batch_size = 256;

local tng_data_path = "datasets/stockmark-ner-wiki/tng_ner.txt";
local val_data_path = "datasets/stockmark-ner-wiki/val_ner.txt";
local tst_data_path = "datasets/stockmark-ner-wiki/tst_ner.txt";

local pretrained_transformer_mismatched = "pretrained_transformer_mismatched";
local transformer_model_name = "cl-tohoku/bert-base-japanese-whole-word-masking";
local bert_hidden_dim = 768;

local dataset_reader = {
    type: "conll2003",
    tag_label: "ner",
    token_indexers: {
        tokens: {
            type: pretrained_transformer_mismatched,
            model_name: transformer_model_name,
        },
    },
};

local model = {
    type: "simple_tagger",
    label_encoding: "BIO",
    calculate_span_f1: true,
    text_field_embedder: {
        token_embedders: {
            tokens: {
                type: pretrained_transformer_mismatched,
                model_name: transformer_model_name,
            },
        },
    },
    encoder: {
        type: "pass_through",
        input_dim: bert_hidden_dim,
    },
};

local tng_data_loader = {
    batch_sampler: {
        type: "bucket",
        batch_size: batch_size,
        sorting_keys: ["tokens"],
        shuffle: true,
    },
};

local val_data_loader = {
    batch_sampler: {
        type: "bucket",
        batch_size: batch_size,
        sorting_keys: ["tokens"],
        shuffle: false,
    },
};

local optimizer = {
    type: "huggingface_adamw",
    lr: 5e-5,
};

local trainer = {
    optimizer: optimizer,
    validation_metric: "-loss",
    num_epochs: num_epochs,
    cuda_device: cuda_device,
    callbacks: [
        {type: "track_epoch_callback"},
        {type: "tensorboard"},
    ],
};

{
    random_seed: seed,
    numpy_seed: seed,
    pytorch_seed: seed,

    train_data_path: tng_data_path,
    validation_data_path: val_data_path,
    test_data_path: tst_data_path,
    datasets_for_vocab_creation: ["train"],
    evaluate_on_test: true,

    dataset_reader: dataset_reader,
    data_loader: tng_data_loader,
    validation_data_loader: val_data_loader,
    model: model,
    trainer: trainer,
}
