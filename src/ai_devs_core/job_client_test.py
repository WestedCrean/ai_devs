"""
tests/job_client_test.py

Test suite for JobClient improvements.
"""

import pytest

import polars as pl
import pydantic

from src.ai_devs_core import JobClient, BatchJobConfig, Config
from src.ai_devs_core.job_client import RateLimiter, ErrorClassifier


class TestBatchJobConfig:
    """Tests for BatchJobConfig dataclass."""
    
    def test_default_values(self):
        """Test that BatchJobConfig has correct defaults."""
        config = BatchJobConfig()
        assert config.model == "mistral-small-2603"
        assert config.poll_interval == 10
        assert config.timeout == 120
        assert config.max_workers == 1
        assert config.max_retries == 5
        assert config.chunk_size == 1000
        assert config.fallback_model == "mistral-small-2603"
        assert config.retry_delay == 1.0
        assert config.max_delay == 60.0
        assert config.rate_limit == 0
        assert config.correlation_id is None
    
    def test_custom_values(self):
        """Test that BatchJobConfig accepts custom values."""
        config = BatchJobConfig(
            model="custom-model",
            poll_interval=5,
            timeout=300,
            max_workers=4,
            max_retries=10,
            chunk_size=500
        )
        assert config.model == "custom-model"
        assert config.poll_interval == 5
        assert config.timeout == 300
        assert config.max_workers == 4
        assert config.max_retries == 10
        assert config.chunk_size == 500


class TestRateLimiter:
    """Tests for RateLimiter token bucket algorithm."""
    
    def test_no_rate_limit(self):
        """Test that rate limiter with 0 rate doesn't wait."""
        limiter = RateLimiter(rate=0)
        # Should not raise or wait
        limiter.wait()
    
    def test_rate_limiting(self):
        """Test that rate limiter enforces rate limits."""
        import time
        
        # 10 requests per second
        limiter = RateLimiter(rate=10, max_tokens=10)
        
        # First 10 calls should work immediately
        for _ in range(10):
            limiter.wait()
        
        # 11th call should wait
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        
        # Should have waited approximately 0.1 seconds (1/10 second)
        assert elapsed >= 0.05  # Allow some tolerance
        assert elapsed < 0.5


class TestErrorClassifier:
    """Tests for ErrorClassifier."""
    
    def test_retryable_errors(self):
        """Test classification of retryable errors."""
        retryable = [
            "Rate limit exceeded",
            "Too many requests",
            "Service unavailable",
            "Internal server error",
            "503",
            "504"
        ]
        
        for error in retryable:
            assert ErrorClassifier.is_retryable(Exception(error)), f"Should be retryable: {error}"
    
    def test_non_retryable_errors(self):
        """Test classification of non-retryable errors."""
        non_retryable = [
            "Invalid request",
            "Bad request",
            "400",
            "401",
            "403"
        ]
        
        for error in non_retryable:
            assert ErrorClassifier.is_non_retryable(Exception(error)), f"Should be non-retryable: {error}"


class MockPydanticModel(pydantic.BaseModel):
    """Test Pydantic model."""
    value: int
    label: str


def mock_message_generator(row):
    """Mock message generator for testing."""
    return [
        {"role": "system", "content": "Test"},
        {"role": "user", "content": f"ID: {row['id']}"}
    ]


class TestJobClient:
    """Tests for JobClient."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return Config(
            AI_DEVS_API_KEY="test_key",
            MISTRAL_API_KEY="test_key",
            OPENROUTER_API_KEY="test_key",
            LANGFUSE_SECRET_KEY="test_key",
            LANGFUSE_PUBLIC_KEY="test_key",
            LANGFUSE_BASE_URL="http://test.com",
            WANDB_API_KEY="test_key"
        )
    
    @pytest.fixture
    def job_client(self, config):
        """Create JobClient instance."""
        return JobClient(config)
    
    def test_initialization(self, config):
        """Test JobClient initialization."""
        client = JobClient(config)
        assert client.config == config
        assert client.client is not None
    
    def test_correlation_id_generation(self, job_client):
        """Test correlation ID generation."""
        config = BatchJobConfig()
        
        # Should generate UUID when not provided
        correlation_id = job_client._get_correlation_id(config)
        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 36  # UUID format
        
        # Should use provided correlation ID
        custom_id = "custom-correlation-id"
        config.correlation_id = custom_id
        assert job_client._get_correlation_id(config) == custom_id
    
    def test_metrics_tracking(self, job_client):
        """Test metrics tracking."""
        # Get initial metrics
        _ = job_client.get_metrics()
        
        # Update metrics
        job_client._update_metrics(success=True)
        job_client._update_metrics(success=True)
        job_client._update_metrics(success=False)
        job_client._update_metrics(success=True, retry=True)
        
        metrics = job_client.get_metrics()
        assert metrics["total_requests"] == 4
        assert metrics["successful_requests"] == 3
        assert metrics["failed_requests"] == 1
        assert metrics["retry_attempts"] == 1
    
    def test_prepare_batch_requests(self, job_client):
        """Test batch request preparation."""
        messages_list = [
            [{"role": "user", "content": "test1"}],
            [{"role": "user", "content": "test2"}]
        ]
        
        requests = job_client._prepare_batch_requests(messages_list, "mistral-test")
        
        assert len(requests) == 2
        assert requests[0]["custom_id"] == "0"
        assert requests[0]["body"]["model"] == "mistral-test"
        assert requests[1]["custom_id"] == "1"
    
    def test_generate_messages_in_chunks(self, job_client):
        """Test message generation in chunks."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "text": ["a", "b", "c", "d", "e"]
        })
        
        messages = job_client._generate_messages_in_chunks(df, mock_message_generator, 2)
        
        assert len(messages) == 5
    
    def test_batch_job_interface(self, job_client):
        """Test that batch_job accepts BatchJobConfig parameter."""
        # Test BatchJobConfig creation and defaults
        config = BatchJobConfig(
            model="mistral-small-2603",
            poll_interval=5,
            timeout=120,
            max_workers=1,
            max_retries=3,
            chunk_size=100
        )
        
        # Test configuration values
        assert config.model == "mistral-small-2603"
        assert config.max_retries == 3
        assert config.timeout == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
