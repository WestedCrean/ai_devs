import polars as pl
import pydantic
import json
import time
import uuid
import threading
import datetime
from typing import List, Dict, Any, Type, Optional, Callable
from loguru import logger
from mistralai.client import Mistral
from mistralai.client.models.batchrequest import BatchRequest
from src.ai_devs_core.config import Config, BatchJobConfig


class RateLimiter:
    """
    Token bucket rate limiter for API throttling.

    Attributes:
        rate: Maximum requests per second (0 = unlimited)
        tokens: Current available tokens
        max_tokens: Maximum tokens (bucket size)
        refill_rate: Tokens added per second
        last_refill: Last time tokens were refilled
    """

    def __init__(self, rate: int = 0, max_tokens: int = 100):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second (0 = unlimited)
            max_tokens: Maximum tokens in bucket
        """
        self.rate = rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def wait(self) -> None:
        """Wait for a token to be available."""
        if self.rate <= 0:
            return  # No rate limiting

        with self.lock:
            # Refill tokens based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.tokens + elapsed * self.rate, self.max_tokens)
            self.last_refill = now

            # If no tokens available, wait
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                self.tokens = 0
                self.last_refill = now + sleep_time
                time.sleep(sleep_time)
            else:
                self.tokens -= 1


class ErrorClassifier:
    """
    Classifies errors for retry logic.

    Attributes:
        retryable_errors: Set of error patterns that can be retried
    """

    RETRYABLE_ERRORS = {
        "rate limit",
        "too many requests",
        "service unavailable",
        "internal server error",
        "gateway timeout",
        "502",
        "503",
        "504",
    }

    NON_RETRYABLE_ERRORS = {
        "invalid request",
        "bad request",
        "400",
        "401",
        "403",
        "404",
        "validation error",
    }

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """
        Check if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error is retryable, False otherwise
        """
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in cls.RETRYABLE_ERRORS)

    @classmethod
    def is_non_retryable(cls, error: Exception) -> bool:
        """
        Check if an error is non-retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error is non-retryable, False otherwise
        """
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in cls.NON_RETRYABLE_ERRORS)


class JobClient:
    """
    Client for processing batch jobs with Mistral API.

    This class provides a high-level interface for Mistral batch jobs with
    observability, error handling, rate limiting, and parallel processing.

    Attributes:
        config: Application configuration
        client: Mistral API client
        rate_limiter: Rate limiter for API throttling
    """

    def __init__(self, config: Config, trace_name: str = None):
        """
        Initialize the JobClient with Mistral API configuration.

        Args:
            config: Configuration containing MISTRAL_API_KEY
        """

        self.config = config
        self.client = Mistral(api_key=config.MISTRAL_API_KEY)
        self.trace_name = trace_name

        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_attempts": 0,
        }
        self._init_langfuse()
        self._metrics_lock = threading.Lock()

    def _init_langfuse(self) -> None:
        """Initialize Langfuse for observability."""
        if self.trace_name is None:
            logger.warning(
                "trace_name parameter not passed to JobClient, Observability disabled"
            )
            self.langfuse = None
            return
        try:
            from langfuse import Langfuse

            self.langfuse = Langfuse(
                public_key=self.config.LANGFUSE_PUBLIC_KEY,
                secret_key=self.config.LANGFUSE_SECRET_KEY,
                base_url=self.config.LANGFUSE_BASE_URL,
            )
            logger.info("Langfuse initialized successfully")
        except ImportError:
            logger.warning("Langfuse not available, observability disabled")
            self.langfuse = None
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            self.langfuse = None

    def _get_correlation_id(self, config: Optional[BatchJobConfig] = None) -> str:
        """Generate or get correlation ID."""
        if config and config.correlation_id:
            return config.correlation_id
        return str(uuid.uuid4())

    def _update_metrics(self, success: bool, retry: bool = False) -> None:
        """Update internal metrics."""
        with self._metrics_lock:
            self._metrics["total_requests"] += 1
            if success:
                self._metrics["successful_requests"] += 1
            else:
                self._metrics["failed_requests"] += 1
            if retry:
                self._metrics["retry_attempts"] += 1

    def _observe_llm_call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        func: Callable,
        *args,
        **kwargs,
    ):
        """
        Observe LLM call with Langfuse.

        Args:
            messages: LLM messages
            model: Model name
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function call
        """
        if not self.langfuse:
            return func(*args, **kwargs)

        try:
            from langfuse import observe

            # Calculate input token count (approximate)
            _ = sum(len(str(msg.get("content", ""))) // 4 for msg in messages)

            @observe(as_type="generation")
            def _wrapped():
                return func(*args, **kwargs)

            result = _wrapped()

            # Update metrics
            self._update_metrics(success=True)

            return result

        except Exception as e:
            logger.error(f"Langfuse observation failed: {e}")
            self._update_metrics(success=False)
            return func(*args, **kwargs)

    def batch_job(
        self,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        task: str,
        message_generator: Callable,
        config: Optional[BatchJobConfig] = None,
        model: Optional[str] = None,
        max_retries: Optional[int] = None,
        poll_interval: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Process a batch job using Mistral API and return results as polars DataFrame.

        Args:
            df: Input DataFrame containing data to process
            schema: Pydantic model defining the expected response schema
            task: Task name/identifier
            message_generator: Function that takes a row and returns messages for API
            config: BatchJobConfig with processing configuration
            model: Mistral model to use (overrides config)
            max_retries: Maximum number of retries (overrides config)
            poll_interval: Seconds between job status polls (overrides config)

        Returns:
            DataFrame with original data plus additional columns from response schema
        """
        # Merge configuration
        if config is None:
            config = BatchJobConfig()

        # Apply overrides
        effective_model = model or config.model
        effective_max_retries = (
            max_retries if max_retries is not None else config.max_retries
        )
        effective_poll_interval = (
            poll_interval if poll_interval is not None else config.poll_interval
        )

        correlation_id = self._get_correlation_id(config)
        logger.info(
            f"Starting batch job for task: {task} (correlation_id: {correlation_id})"
        )
        logger.info(f"Input DataFrame shape: {df.shape}")

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate=config.rate_limit)

        # Add index column if not present
        if "id" not in df.columns:
            df = df.with_row_index("id")

        # Generate messages for each row in chunks
        messages_list = self._generate_messages_in_chunks(
            df, message_generator, config.chunk_size
        )

        logger.info(f"Generated {len(messages_list)} message sets")

        # Process in parallel if configured
        if config.max_workers > 1:
            return self._process_parallel(
                df,
                schema,
                messages_list,
                message_generator,
                effective_model,
                effective_max_retries,
                correlation_id,
                config.timeout,
            )

        # Process sequentially with retry logic
        return self._process_with_retry(
            df,
            schema,
            messages_list,
            message_generator,
            effective_model,
            effective_max_retries,
            effective_poll_interval,
            correlation_id,
            config.timeout,
        )

    def _generate_messages_in_chunks(
        self, df: pl.DataFrame, message_generator: Callable, chunk_size: int
    ) -> List[List[Dict[str, str]]]:
        """
        Generate messages for each row, processing in chunks for memory efficiency.

        Args:
            df: Input DataFrame
            message_generator: Function to generate messages
            chunk_size: Size of chunks to process

        Returns:
            List of message lists
        """
        messages_list = []

        # Process in chunks to manage memory
        for i in range(0, len(df), chunk_size):
            chunk = df.slice(i, min(chunk_size, len(df) - i))

            for row in chunk.iter_rows(named=True):
                messages = message_generator(row)
                messages_list.append(messages)

            # Log progress
            if i > 0 and i % (chunk_size * 10) == 0:
                logger.info(f"Processed {i} rows...")

        return messages_list

    def _get_available_models(self) -> dict:
        logger.info(f"{type(self.client.models.list())}")
        return self.client.models.list().data

    def _process_with_retry(
        self,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        messages_list: List[List[Dict[str, str]]],
        message_generator: Callable,
        model: str,
        max_retries: int,
        poll_interval: int,
        correlation_id: str,
        timeout: int,
    ) -> pl.DataFrame:
        """
        Process messages with retry logic.

        Args:
            df: Input DataFrame
            schema: Response schema
            messages_list: List of messages
            message_generator: Message generator function
            model: Model to use
            max_retries: Maximum retries
            poll_interval: Poll interval
            correlation_id: Correlation ID for tracing
            timeout: Maximum time to wait

        Returns:
            DataFrame with results
        """
        if self.langfuse:
            trace = self.langfuse.trace(
                name=self.trace_name,
                metadata={"task": "batch_processing", "correlation_id": correlation_id},
            )
        else:
            trace = None

        try:
            with (
                self.langfuse.trace(name=self.trace_name)
                if self.langfuse
                else nullcontext()
            ):
                # Prepare batch requests
                batch_requests = self._prepare_batch_requests(df, messages_list, model)

                # Create batch job
                try:
                    batch_job = self._create_batch_job(batch_requests, model)
                    logger.info(f"Batch job created with ID: {batch_job.id}")

                    if trace:
                        trace.update(metadata={"batch_job_id": batch_job.id})
                except Exception as e:
                    logger.error(f"Failed to create batch job: {e}")

                    if (
                        "Please check the available models via the /v1/models endpoint."
                        in str(e)
                    ):
                        # show models
                        logger.info(
                            f"Available models: {[f"{m.id}" for m in self._get_available_models()]}"
                        )
                    # Fallback to sequential processing
                    return self._process_sequential_with_retry(
                        df,
                        schema,
                        messages_list,
                        message_generator,
                        model,
                        max_retries,
                        correlation_id,
                        timeout,
                    )

                # Wait for job completion
                return self._wait_for_job_completion(
                    batch_job.id,
                    df,
                    schema,
                    messages_list,
                    poll_interval,
                    correlation_id,
                    timeout,
                )

        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            if trace:
                trace.update(status="failed", error=str(e))
            raise
        finally:
            if trace:
                trace.update(
                    metadata={**trace.get_metadata(), "metrics": self._metrics}
                )

    def _process_parallel(
        self,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        messages_list: List[List[Dict[str, str]]],
        message_generator: Callable,
        model: str,
        max_retries: int,
        correlation_id: str,
        timeout: int,
    ) -> pl.DataFrame:
        """
        Process messages in parallel using ThreadPoolExecutor.

        Args:
            df: Input DataFrame
            schema: Response schema
            messages_list: List of messages
            message_generator: Message generator function
            model: Model to use
            max_retries: Maximum retries
            correlation_id: Correlation ID for tracing
            timeout: Maximum time to wait

        Returns:
            DataFrame with results
        """
        import concurrent.futures

        logger.info(
            f"Processing {len(messages_list)} messages with {min(len(messages_list), self.config.max_workers)} workers"
        )

        results_data = [None] * len(messages_list)

        def process_single(i: int, messages: List[Dict[str, str]]) -> Dict[str, Any]:
            """Process a single message set."""
            try:
                # Wait for rate limit
                self.rate_limiter.wait()

                # Make API call with retry
                for attempt in range(max_retries + 1):
                    try:
                        response = self.client.chat.complete(
                            model=model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            max_tokens=256,
                            temperature=0,
                        )

                        response_content = response.choices[0].message.content
                        logger.debug(f"Raw LLM response: {response_content}")
                        response_dict = json.loads(response_content)

                        # Validate against schema
                        validated_response = schema(**response_dict)

                        result = validated_response.model_dump()
                        result["_batch_id"] = i
                        result["_success"] = True

                        self._update_metrics(success=True)
                        return result

                    except json.JSONDecodeError as e:
                        # Non-retryable error
                        if ErrorClassifier.is_non_retryable(e):
                            raise
                        logger.warning(
                            f"JSON decode error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        if attempt < max_retries:
                            time.sleep(min(2**attempt, 30))
                        else:
                            raise

                    except Exception as e:
                        if ErrorClassifier.is_retryable(e):
                            if attempt < max_retries:
                                delay = min(2**attempt + (0.1 * (attempt + 1)), 60)
                                logger.warning(
                                    f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                                )
                                self._update_metrics(success=False, retry=True)
                                time.sleep(delay)
                                continue

                        logger.error(f"Non-retryable error processing row {i}: {e}")
                        result = {"_batch_id": i, "_success": False, "_error": str(e)}
                        self._update_metrics(success=False)
                        return result

            except Exception as e:
                logger.error(f"Failed to process row {i}: {e}")
                return {"_batch_id": i, "_success": False, "_error": str(e)}

        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(messages_list), 4)
        ) as executor:
            future_to_idx = {
                executor.submit(process_single, i, messages): i
                for i, messages in enumerate(messages_list)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_data[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    results_data[idx] = {
                        "_batch_id": idx,
                        "_success": False,
                        "_error": str(e),
                    }

        # Ensure results_data is not empty
        if not results_data:
            logger.warning("No results generated from batch job")
            results_data = [
                {"_batch_id": i, "_success": False, "_error": "No results generated"}
                for i in range(len(df))
            ]

        # Create results DataFrame
        results_df = pl.DataFrame(results_data)

        # Merge with original DataFrame
        if "id" in df.columns:
            merged_df = df.join(
                results_df, left_on="id", right_on="_batch_id", how="left"
            )
            if "_batch_id" in merged_df.columns:
                merged_df = merged_df.drop("_batch_id")
        else:
            merged_df = pl.concat([df, results_df], how="horizontal")

        return merged_df

    def _process_sequential_with_retry(
        self,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        messages_list: List[List[Dict[str, str]]],
        message_generator: Callable,
        model: str,
        max_retries: int,
        correlation_id: str,
        timeout: int,
    ) -> pl.DataFrame:
        """
        Fallback: Process messages sequentially with retry logic.

        Args:
            df: Input DataFrame
            schema: Response schema
            messages_list: List of messages
            message_generator: Message generator function
            model: Model to use
            max_retries: Maximum retries
            correlation_id: Correlation ID for tracing
            timeout: Maximum time to wait

        Returns:
            DataFrame with results
        """
        logger.warning("Falling back to sequential processing with retry...")

        results_data = []

        for i, messages in enumerate(messages_list):
            try:
                # Wait for rate limit
                self.rate_limiter.wait()

                # Make API call with retry
                for attempt in range(max_retries + 1):
                    try:
                        response = self.client.chat.complete(
                            model=model,
                            messages=messages,
                            response_format={"type": "json_object"},
                            max_tokens=256,
                            temperature=0,
                        )

                        response_content = response.choices[0].message.content
                        response_dict = json.loads(response_content)

                        # Validate against schema
                        validated_response = schema(**response_dict)
                        response_data = validated_response.model_dump()

                        # Add metadata - use the actual id from the row if available
                        if i < len(df) and "id" in df.columns:
                            row_id = df["id"].to_list()[i]
                            response_data["_batch_id"] = row_id
                        else:
                            response_data["_batch_id"] = i
                        response_data["_success"] = True

                        results_data.append(response_data)
                        logger.info(f"Processed row {i} sequentially")
                        self._update_metrics(success=True)
                        break

                    except json.JSONDecodeError as e:
                        # Non-retryable error
                        if ErrorClassifier.is_non_retryable(e):
                            raise
                        logger.warning(
                            f"JSON decode error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        if attempt < max_retries:
                            time.sleep(min(2**attempt, 30))
                        else:
                            raise

                    except Exception as e:
                        if ErrorClassifier.is_retryable(e):
                            if attempt < max_retries:
                                delay = min(2**attempt + (0.1 * (attempt + 1)), 60)
                                logger.warning(
                                    f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                                )
                                self._update_metrics(success=False, retry=True)
                                time.sleep(delay)
                                continue

                        logger.error(f"Non-retryable error processing row {i}: {e}")
                        self._update_metrics(success=False)
                        results_data.append(
                            {"_batch_id": i, "_success": False, "_error": str(e)}
                        )
                        break

            except Exception as e:
                logger.error(f"Failed to process row {i}: {e}")
                results_data.append(
                    {"_batch_id": i, "_success": False, "_error": str(e)}
                )

        # Create results DataFrame
        # Ensure results_data is not empty
        if not results_data:
            logger.warning("No results generated from sequential processing")
            results_data = [
                {"_batch_id": i, "_success": False, "_error": "No results generated"}
                for i in range(len(df))
            ]

        results_df = pl.DataFrame(results_data)

        # Merge with original DataFrame
        if "id" in df.columns:
            merged_df = df.join(
                results_df, left_on="id", right_on="_batch_id", how="left"
            )
            if "_batch_id" in merged_df.columns:
                merged_df = merged_df.drop("_batch_id")
        else:
            merged_df = pl.concat([df, results_df], how="horizontal")

        return merged_df

    def _prepare_batch_requests(
        self, df: pl.DataFrame, messages_list: List[List[Dict[str, str]]], model: str
    ) -> List[BatchRequest]:
        """
        Prepare batch requests for Mistral API.

        Args:
            df: Original DataFrame containing the id column
            messages_list: List of message lists
            model: Model to use

        Returns:
            List of BatchRequest objects
        """
        batch_requests = []
        for i, messages in enumerate(messages_list):
            # Use the actual id from the DataFrame row, not the loop index
            row_id = str(df["id"].to_list()[i])
            batch_requests.append(
                BatchRequest(
                    custom_id=row_id,
                    body={
                        "model": model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "max_tokens": 256,
                        "temperature": 0,
                    },
                )
            )
        return batch_requests

    def _create_batch_job(self, batch_requests: List[BatchRequest], model: str):
        """
        Create a batch job.

        Args:
            batch_requests: List of batch requests
            model: Model to use

        Returns:
            Batch job object
        """
        logger.info(f"Launching a batch job, model={model}")

        # Save batch requests to JSON file (convert BatchRequest objects to dicts for serialization)
        from pathlib import Path

        batch_dir = Path("./data/batch_jobs")
        batch_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = batch_dir / f"batch_requests_{timestamp}.json"

        # Convert BatchRequest objects to dictionaries for JSON serialization
        batch_requests_dicts = [request.model_dump() for request in batch_requests]
        with open(filename, "w") as f:
            json.dump(batch_requests_dicts, f, indent=2)

        logger.info(f"Saved batch requests to {filename}")

        return self.client.batch.jobs.create(
            endpoint="/v1/chat/completions", model=model, requests=batch_requests
        )

    def _wait_for_job_completion(
        self,
        job_id: str,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        messages_list: List[List[Dict[str, str]]],
        poll_interval: int,
        correlation_id: str,
        timeout: int,
    ) -> pl.DataFrame:
        """Wait for batch job to complete and process results."""
        logger.info(f"Waiting for batch job {job_id} to complete...")

        start_time = time.time()

        while True:
            try:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.error("Timeout waiting for job completion")
                    return self._process_sequential_with_retry(
                        df,
                        schema,
                        messages_list,
                        None,
                        "mistral-small-2603",
                        5,
                        correlation_id,
                        timeout,
                    )

                job_status = self.client.batch.jobs.get(job_id=job_id)
                logger.info(f"Job status: {job_status.status}")

                if job_status.status == "SUCCESS":
                    logger.info("Job completed successfully!")
                    return self._process_batch_results(
                        job_id, df, schema, messages_list, correlation_id
                    )
                elif job_status.status == "failed":
                    logger.error(
                        f"Batch job failed: {getattr(job_status, 'error', 'Unknown error')}"
                    )
                    return self._process_sequential_with_retry(
                        df,
                        schema,
                        messages_list,
                        None,
                        "mistral-small-2603",
                        5,
                        correlation_id,
                        timeout,
                    )

                # Wait before polling again
                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error checking job status: {e}")
                time.sleep(poll_interval)

    def _process_batch_results(
        self,
        job_id: str,
        df: pl.DataFrame,
        schema: Type[pydantic.BaseModel],
        messages_list: List[List[Dict[str, str]]],
        correlation_id: str,
    ) -> pl.DataFrame:
        """Process batch job results and merge with original DataFrame."""
        try:
            logger.info("Retrieving batch results...")
            # Get the job object which should contain results
            job_obj = self.client.batch.jobs.get(job_id=job_id)

            # Check if results are available in the job object
            if hasattr(job_obj, "outputs") and job_obj.outputs:
                batch_results = job_obj.outputs
            else:
                logger.error("No results found in batch job object")
                raise AttributeError("Batch job results not available")

            # Process results
            results_data = []
            for i, result in enumerate(batch_results):
                try:
                    # Access the response content from the nested structure
                    # Result is a dict with 'response' -> 'body' -> 'choices' -> 'message' -> 'content'
                    response_content = result['response']['body']['choices'][0]['message']['content']
                    response_dict = json.loads(response_content)

                    # Validate against schema
                    validated_response = schema(**response_dict)
                    response_data = validated_response.model_dump()

                    # Add metadata - use custom_id from the batch result to match with original DataFrame
                    custom_id = result.get('custom_id', str(i))
                    # Convert custom_id to int to match the id column type in the original DataFrame
                    try:
                        response_data["_batch_id"] = int(custom_id)
                    except ValueError:
                        response_data["_batch_id"] = custom_id
                    response_data["_success"] = True

                    results_data.append(response_data)

                except Exception as e:
                    logger.error(f"Error processing result {i}: {e}")
                    custom_id = result.get('custom_id', str(i))
                    # Convert custom_id to int to match the id column type in the original DataFrame
                    try:
                        results_data.append(
                            {"_batch_id": int(custom_id), "_success": False, "_error": str(e)}
                        )
                    except ValueError:
                        results_data.append(
                            {"_batch_id": custom_id, "_success": False, "_error": str(e)}
                        )

            # Ensure results_data is not empty - add a placeholder if needed
            if not results_data:
                logger.warning("No results returned from batch job")
                # Use original df rows as placeholders
                results_data = [
                    {"_batch_id": i, "_success": False, "_error": "No results returned"}
                    for i in range(len(df))
                ]

            # Create results DataFrame
            results_df = pl.DataFrame(results_data)

            # Merge with original DataFrame
            if "id" in df.columns:
                merged_df = df.join(
                    results_df, left_on="id", right_on="_batch_id", how="left"
                )
                if "_batch_id" in merged_df.columns:
                    merged_df = merged_df.drop("_batch_id")
            else:
                # If no id column, just concatenate
                merged_df = pl.concat([df, results_df], how="horizontal")

            return merged_df

        except Exception as e:
            logger.error(f"Error processing batch results: {e}")
            return self._process_sequential_with_retry(
                df,
                schema,
                messages_list,
                None,
                "mistral-small-2603",
                5,
                correlation_id,
                120,
            )

    def get_metrics(self) -> Dict[str, int]:
        """Get processing metrics."""
        with self._metrics_lock:
            return self._metrics.copy()


class nullcontext:
    """Null context manager for compatibility."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass
