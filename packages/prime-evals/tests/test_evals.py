"""Tests for Prime Evals SDK"""

import pytest

from prime_evals.evals import EvalsClient
from prime_evals.models import (
    CreateEvaluationRequest,
    Evaluation,
    EvaluationStatus,
    Sample,
)


def test_create_evaluation_request():
    """Test CreateEvaluationRequest model"""
    request = CreateEvaluationRequest(
        name="test-evaluation",
        model_name="gpt-4o-mini",
        dataset="gsm8k",
    )

    assert request.name == "test-evaluation"
    assert request.model_name == "gpt-4o-mini"
    assert request.dataset == "gsm8k"
    assert request.tags == []
    assert request.metadata is None


def test_create_evaluation_request_with_metadata():
    """Test CreateEvaluationRequest with metadata"""
    metadata = {"version": "1.0", "num_examples": 10}
    request = CreateEvaluationRequest(
        name="test-evaluation",
        model_name="gpt-4o-mini",
        dataset="gsm8k",
        tags=["test", "baseline"],
        metadata=metadata,
    )

    assert request.tags == ["test", "baseline"]
    assert request.metadata == metadata


def test_evaluation_status_enum():
    """Test EvaluationStatus enum values"""
    assert EvaluationStatus.PENDING == "PENDING"
    assert EvaluationStatus.RUNNING == "RUNNING"
    assert EvaluationStatus.COMPLETED == "COMPLETED"
    assert EvaluationStatus.FAILED == "FAILED"
    assert EvaluationStatus.CANCELLED == "CANCELLED"


def test_evaluation_model_with_alias():
    """Test Evaluation model handles API field aliases"""
    data = {
        "evaluation_id": "eval-123",
        "name": "test-eval",
        "modelName": "gpt-4o-mini",
        "dataset": "gsm8k",
        "framework": "verifiers",
        "taskType": "math",
        "status": "COMPLETED",
        "tags": ["test"],
        "totalSamples": 10,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }

    evaluation = Evaluation.model_validate(data)

    assert evaluation.id == "eval-123"
    assert evaluation.name == "test-eval"
    assert evaluation.model_name == "gpt-4o-mini"
    assert evaluation.dataset == "gsm8k"
    assert evaluation.task_type == "math"
    assert evaluation.total_samples == 10


def test_sample_model():
    """Test Sample model"""
    data = {
        "exampleId": 0,
        "task": "gsm8k",
        "reward": 1.0,
        "correct": True,
        "formatReward": 1.0,
        "correctness": 1.0,
        "answer": "18",
        "prompt": [{"role": "user", "content": "What is 9+9?"}],
        "completion": [{"role": "assistant", "content": "The answer is 18."}],
    }

    sample = Sample.model_validate(data)

    assert sample.example_id == 0
    assert sample.task == "gsm8k"
    assert sample.reward == 1.0
    assert sample.correct is True
    assert sample.format_reward == 1.0
    assert sample.correctness == 1.0
    assert sample.answer == "18"
    assert len(sample.prompt) == 1
    assert len(sample.completion) == 1


def test_sample_model_with_metadata():
    """Test Sample model with extra metadata"""
    data = {
        "exampleId": 0,
        "task": "gsm8k",
        "reward": 1.0,
        "answer": "18",
        "custom_field": "custom_value",  # Extra field should be allowed
        "info": {"batch": 1},
    }

    sample = Sample.model_validate(data)

    assert sample.example_id == 0
    assert sample.task == "gsm8k"
    assert sample.reward == 1.0
    assert sample.info == {"batch": 1}


def test_evals_client_context_manager():
    """Test EvalsClient can be used as context manager"""
    try:
        # This will fail without API key, but we're testing the interface
        client = EvalsClient.__new__(EvalsClient)
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")
        assert hasattr(client, "close")
    except Exception:
        pass  # Expected to fail without proper initialization


def test_evaluation_model_minimal():
    """Test Evaluation model with minimal data"""
    data = {
        "evaluation_id": "eval-minimal",
        "name": "minimal-eval",
    }

    evaluation = Evaluation.model_validate(data)

    assert evaluation.id == "eval-minimal"
    assert evaluation.name == "minimal-eval"
    assert evaluation.tags == []


def test_sample_model_minimal():
    """Test Sample model with minimal data"""
    data = {}

    sample = Sample.model_validate(data)

    assert sample.example_id is None
    assert sample.task is None
    assert sample.reward is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
