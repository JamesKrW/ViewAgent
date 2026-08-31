"""Selective generated-token decoding on the model-adapter boundary."""

from __future__ import annotations

from vagen_agent.models import ModelAdapter


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 1,
            "<eos>": 2,
            "<click>": 3,
            "<task-stop>": 4,
            "5": 5,
        }
        self.special_tokens_map = {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "additional_special_tokens": ["<click>", "<task-stop>"],
        }

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse[value] for value in ids)


def test_base_adapter_skips_controls_but_keeps_task_special_tokens() -> None:
    model = ModelAdapter(FakeTokenizer())
    assert model.decode_generated([1, 3, 5, 2]) == "<click>5"


def test_model_family_can_name_an_extra_control_role() -> None:
    class FamilyTokenizer(FakeTokenizer):
        task_stop_token = "<task-stop>"

    class FamilyAdapter(ModelAdapter):
        generation_control_token_fields = (
            *ModelAdapter.generation_control_token_fields,
            "task_stop_token",
        )

    model = FamilyAdapter(FamilyTokenizer())
    assert model.decode_generated([3, 5, 4]) == "<click>5"
    assert model.decode_generated(
        [3, 5, 4], preserve_special_tokens=["<task-stop>"]
    ) == "<click>5<task-stop>"


def main() -> None:
    test_base_adapter_skips_controls_but_keeps_task_special_tokens()
    test_model_family_can_name_an_extra_control_role()
    print("PASS model adapter selective decoding")


if __name__ == "__main__":
    main()
