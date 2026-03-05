"""Tests for synthetic data generators."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from evalkit.generators.synthetic import SyntheticGenerator
from evalkit.generators.templates import (
    GenerationStrategy,
    STRATEGY_TEMPLATES,
    render_template,
)


class TestGenerationStrategy:
    """Tests for generation strategy enum."""

    def test_strategy_values(self) -> None:
        assert GenerationStrategy.STANDARD.value == "standard"
        assert GenerationStrategy.ADVERSARIAL.value == "adversarial"
        assert GenerationStrategy.EDGE_CASE.value == "edge_case"
        assert GenerationStrategy.DISTRIBUTION_MATCHING.value == "distribution_matching"


class TestRenderTemplate:
    """Tests for template rendering."""

    @pytest.mark.parametrize("strategy", list(GenerationStrategy))
    def test_template_renders_topic(self, strategy: GenerationStrategy) -> None:
        result = render_template(strategy, "machine learning", 10)
        assert "machine learning" in result
        assert "10" in result

    def test_extra_instructions_included(self) -> None:
        result = render_template(
            GenerationStrategy.STANDARD,
            "python",
            5,
            extra_instructions="Focus on data structures.",
        )
        assert "Focus on data structures." in result

    def test_all_strategies_have_templates(self) -> None:
        for strategy in GenerationStrategy:
            assert strategy in STRATEGY_TEMPLATES


class TestSyntheticGenerator:
    """Tests for the SyntheticGenerator class."""

    def test_init_defaults(self) -> None:
        gen = SyntheticGenerator()
        assert gen.generator_id == "synthetic-default"
        assert gen.strategy == GenerationStrategy.STANDARD

    def test_init_custom(self) -> None:
        gen = SyntheticGenerator(
            generator_id="adv-gen",
            strategy=GenerationStrategy.ADVERSARIAL,
        )
        assert gen.generator_id == "adv-gen"
        assert gen.strategy == GenerationStrategy.ADVERSARIAL

    def test_parse_clean_json(self) -> None:
        raw = json.dumps([
            {"input": "What is ML?", "difficulty": "easy"},
            {"input": "Explain backpropagation", "difficulty": "hard"},
        ])
        result = SyntheticGenerator._parse_response(raw)
        assert len(result) == 2
        assert result[0]["input"] == "What is ML?"

    def test_parse_json_in_code_block(self) -> None:
        raw = '```json\n[{"input": "test"}]\n```'
        result = SyntheticGenerator._parse_response(raw)
        assert len(result) == 1

    def test_parse_filters_items_without_input(self) -> None:
        raw = json.dumps([
            {"input": "valid", "category": "common"},
            {"no_input_key": "invalid"},
        ])
        result = SyntheticGenerator._parse_response(raw)
        assert len(result) == 1

    def test_parse_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            SyntheticGenerator._parse_response("not json")

    def test_parse_non_array(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON array"):
            SyntheticGenerator._parse_response('{"key": "value"}')

    def test_deduplicate(self) -> None:
        items = [
            {"input": "What is Python?"},
            {"input": "what is python?"},  # case-insensitive duplicate
            {"input": "Explain decorators"},
        ]
        result = SyntheticGenerator._deduplicate(items)
        assert len(result) == 2

    def test_deduplicate_empty_input(self) -> None:
        items = [
            {"input": ""},
            {"input": "valid"},
        ]
        result = SyntheticGenerator._deduplicate(items)
        assert len(result) == 1
        assert result[0]["input"] == "valid"

    @patch.object(SyntheticGenerator, "_call_llm")
    def test_generate_calls_llm(self, mock_call: object) -> None:
        from unittest.mock import MagicMock

        assert isinstance(mock_call, MagicMock)
        mock_call.return_value = json.dumps([
            {"input": "Q1", "difficulty": "easy"},
            {"input": "Q2", "difficulty": "medium"},
            {"input": "Q3", "difficulty": "hard"},
        ])
        gen = SyntheticGenerator()
        results = gen.generate("machine learning", count=3)
        assert len(results) == 3
        mock_call.assert_called_once()

    @patch.object(SyntheticGenerator, "_call_llm")
    def test_generate_deduplicates(self, mock_call: object) -> None:
        from unittest.mock import MagicMock

        assert isinstance(mock_call, MagicMock)
        mock_call.return_value = json.dumps([
            {"input": "Q1"},
            {"input": "q1"},  # duplicate
            {"input": "Q2"},
        ])
        gen = SyntheticGenerator()
        results = gen.generate("topic", count=3)
        assert len(results) == 2

    def test_repr(self) -> None:
        gen = SyntheticGenerator(generator_id="test-gen")
        assert "SyntheticGenerator" in repr(gen)
        assert "test-gen" in repr(gen)
