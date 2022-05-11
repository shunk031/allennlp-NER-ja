from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits


@Model.register("cost_sensitive_simple_tagger")
class CostSensitiveSimpleTagger(SimpleTagger):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            calculate_span_f1,
            label_encoding,
            label_namespace,
            verbose_metrics,
            initializer,
            **kwargs
        )

        self.class_weight = self.calculate_class_weight(
            vocab=vocab, label_namespace=label_namespace
        )

    def calculate_class_weight(self, vocab: Vocabulary, label_namespace: str):

        if vocab._retained_counter:
            label_counter = vocab._retained_counter[label_namespace]
            bincount = np.array(list(label_counter.values()))
            n_samples = bincount.sum()
            n_classes = len(bincount)
            class_weight = n_samples / (n_classes * bincount)

        else:
            raise ValueError(
                "Class weight cannot be calculated since dataset "
                "instances were not used for its construction."
            )
        breakpoint()  # TODO: check the order of the class_weight
        device = next(self.parameters()).device
        return torch.FloatTensor(class_weight, device=device)

    def forward(
        self,
        tokens: TextFieldTensors,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index(
                    "O", namespace=self.label_namespace
                )
                tag_mask = mask & (tags != o_tag_index)
            else:
                tag_mask = mask
            breakpoint()
            loss = sequence_cross_entropy_with_logits(logits, tags, tag_mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask)
            if self.calculate_span_f1:
                self._f1_metric(logits, tags, mask)  # type: ignore
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict
