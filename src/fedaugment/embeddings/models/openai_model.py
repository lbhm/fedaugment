import asyncio
import atexit
import os
import uuid
from asyncio import Queue, Task
from collections.abc import Awaitable, Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast, get_args, overload

import loguru
import numpy as np
import orjson
import tiktoken
from loguru import logger
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    AsyncOpenAI,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types import Batch
from openai.types import EmbeddingModel as OpenAIEmbeddingModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm

from fedaugment.types import FloatArray
from fedaugment.utils import RateLimiter

from .embedding_model import EmbeddingModel

T = TypeVar("T")
retry_exceptions = (
    APIConnectionError,
    APIResponseValidationError,
    InternalServerError,
    RateLimitError,
)


def before_sleep_loguru(
    logger: "loguru.Logger", log_level: int | str, fn_name: str = "<unknown>"
) -> Callable[[RetryCallState], None]:
    """Before sleep strategy that logs the retry to loguru."""

    def log_it(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            raise RuntimeError("log_it() called before outcome was set")
        if retry_state.next_action is None:
            raise RuntimeError("log_it() called before next_action was set")

        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            verb, value = "raised", f"{ex.__class__.__name__}: {ex}"
        else:
            verb, value = "returned", retry_state.outcome.result()

        logger.log(
            log_level,
            "Retrying {} in {} seconds as it {} {}.",
            fn_name,
            retry_state.next_action.sleep,
            verb,
            value,
        )

    return log_it


class OpenAIBase(EmbeddingModel):
    """Base class for OpenAI embedding models."""

    VALID_MODELS = get_args(OpenAIEmbeddingModel)
    ENCODING = (
        "cl100k_base"  # See https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    )
    EMBEDDING_DIMS = MappingProxyType(
        {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
    )
    MAX_SEQ_LENGHTS = MappingProxyType(
        {
            "text-embedding-ada-002": 8192,
            "text-embedding-3-small": 8192,
            "text-embedding-3-large": 8192,
        }
    )
    MAX_PROMPTS_PER_REQUEST = 2048  # As per the API docs
    MAX_TOKENS_PER_REQUEST = 300_000  # As per the API error message
    TOKENIZER_BATCH_SIZE = 100

    def __init__(
        self, model_id: str, alias: str | None = None, n_tokenizer_threads: int | None = None
    ) -> None:
        super().__init__(model_id, alias)
        if self.model_id not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model '{self.model_id}'. Valid models are: {self.VALID_MODELS}"
            )

        self.max_seq_length = self.MAX_SEQ_LENGHTS[self.model_id]
        self.embedding_dim = self.EMBEDDING_DIMS[self.model_id]
        self.truncate_dim: int | None = None

        self.tokenizer = tiktoken.get_encoding(self.ENCODING)
        self._executor: ThreadPoolExecutor | None = None
        if n_tokenizer_threads is None or n_tokenizer_threads > 1:
            self._executor = ThreadPoolExecutor(n_tokenizer_threads or os.cpu_count())
            atexit.register(self._executor.shutdown, wait=True)

    def tokenize(self, prompts: Iterable[str], max_length: int | None = None) -> list[list[int]]:
        def process_batch(batch: list[str]) -> list[list[int]]:
            return [self.tokenizer.encode(p, disallowed_special=())[:max_length] for p in batch]

        prompts = list(prompts)
        token_iter: Iterator[list[int]]
        if self._executor is not None:
            batches = self._get_prompt_batches(prompts, self.TOKENIZER_BATCH_SIZE)
            batch_iter = self._executor.map(
                process_batch, batches, chunksize=self.TOKENIZER_BATCH_SIZE
            )
            token_iter = chain.from_iterable(batch_iter)
        else:
            token_iter = (
                self.tokenizer.encode(p, disallowed_special=())[:max_length] for p in prompts
            )

        return list(
            tqdm(
                token_iter,
                desc="Tokenizing prompts",
                total=len(prompts),
                leave=False,
                unit="prompt",
                mininterval=1.0,
                dynamic_ncols=True,
            )
        )

    @overload
    def count_tokens(self, prompts: Iterable[str]) -> int: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[True]) -> list[int]: ...

    @overload
    def count_tokens(self, prompts: Iterable[str], per_prompt: Literal[False]) -> int: ...

    def count_tokens(self, prompts: Iterable[str], per_prompt: bool = False) -> int | list[int]:
        def process_batch(batch: list[str]) -> list[int]:
            return [len(self.tokenizer.encode_to_numpy(p, disallowed_special=())) for p in batch]

        prompts = list(prompts)
        count_iter: Iterator[int]
        if self._executor is not None:
            batches = self._get_prompt_batches(prompts, self.TOKENIZER_BATCH_SIZE)
            batch_iter = self._executor.map(
                process_batch, batches, chunksize=self.TOKENIZER_BATCH_SIZE
            )
            count_iter = chain.from_iterable(batch_iter)
        else:
            count_iter = (
                len(self.tokenizer.encode_to_numpy(p, disallowed_special=())) for p in prompts
            )

        wrapped_iter = tqdm(
            count_iter,
            desc="Counting tokens",
            total=len(prompts),
            leave=False,
            unit="prompt",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        return list(wrapped_iter) if per_prompt else sum(wrapped_iter)

    def _get_tokenized_batches(self, prompts: list[str]) -> list[list[list[int]]]:
        """Tokenize and split prompts into batches while respecting API limits.

        Args:
            prompts: A list of prompts.

        Returns:
            A list of batches, where each batch is a list of tokenized prompts.
        """
        tokenized_prompts = self.tokenize(prompts, max_length=self.max_seq_length)

        batches = []
        current_batch: list[list[int]] = []
        n_prompts = 0
        n_tokens = 0

        for tokens in tokenized_prompts:
            if (
                n_prompts + 1 > self.MAX_PROMPTS_PER_REQUEST
                or n_tokens + len(tokens) > self.MAX_TOKENS_PER_REQUEST
            ):
                batches.append(current_batch)
                current_batch = [tokens]
                n_prompts = 1
                n_tokens = len(tokens)
            else:
                current_batch.append(tokens)
                n_prompts += 1
                n_tokens += len(tokens)

        if current_batch:
            batches.append(current_batch)

        return batches


class OpenAIModel(OpenAIBase):
    """OpenAI model using the on-demand API with synchronous calls.

    This class is intended for testing purposes and Jupyter notebooks, where async calls
    may not work due to nested event loops. It is slower and more expensive than the other
    OpenAIModel classes.
    """

    def __init__(
        self, model_id: str, alias: str | None = None, n_tokenizer_threads: int | None = None
    ) -> None:
        super().__init__(model_id, alias, n_tokenizer_threads)
        self.client = OpenAI()

    def embed(
        self, prompt_ids: list[str], prompts: list[str], truncate_dim: int | None = None
    ) -> tuple[list[str], FloatArray]:
        self.truncate_dim = truncate_dim
        batches = self._get_tokenized_batches(prompts)
        logger.info("Converted {} prompts into {} batch requests.", len(prompts), len(batches))

        prompt_iter = tqdm(
            chain.from_iterable(self._get_embedding(batch) for batch in batches),
            desc="Embedding prompts",
            total=len(prompts),
            leave=False,
            unit="prompt",
            mininterval=1.0,
            dynamic_ncols=True,
        )

        embeddings = np.vstack(list(prompt_iter), dtype=np.float32)
        logger.debug("Stacked results into array of shape {}.", embeddings.shape)

        if len(embeddings) != len(prompt_ids):
            raise ValueError(
                f"Expected {len(prompt_ids)} embeddings, but got {len(embeddings)}. "
                "This may indicate an issue with the OpenAI API response."
            )

        return prompt_ids, embeddings

    @retry(
        retry=retry_if_exception_type(retry_exceptions),
        wait=wait_random_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_loguru(logger, "DEBUG", "_get_embedding"),
    )
    def _get_embedding(self, text: list[list[int]]) -> list[list[float]]:
        """Call the OpenAI API for a batch of texts."""
        kwargs: dict[str, Any] = {"input": text, "model": self.model_id}
        if self.truncate_dim is not None:
            kwargs["dimensions"] = self.truncate_dim
        response = self.client.embeddings.create(**kwargs)
        logger.debug(
            "OpenAI generated {} tokens with model {}.",
            response.usage.total_tokens,
            response.model,
        )
        return [d.embedding for d in response.data]


class AsyncOpenAIBase(OpenAIBase):
    """Base class for OpenAI embedding models using async API calls.

    Children of this class do not work in Jupyter notebooks due to a nested event loop.
    """

    def __init__(
        self, model_id: str, alias: str | None = None, n_tokenizer_threads: int | None = None
    ) -> None:
        super().__init__(model_id, alias, n_tokenizer_threads)
        self.client = AsyncOpenAI()

        self._event_loop = asyncio.get_event_loop()

    @retry(
        retry=retry_if_exception_type(retry_exceptions),
        wait=wait_random_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_loguru(logger, "DEBUG", "_api_request"),
    )
    async def _api_request(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Make an async API request and handle exceptions."""
        return await fn(*args, **kwargs)


class AsyncOpenAIModel(AsyncOpenAIBase):
    """OpenAI model using the on-demand API with async calls."""

    REQUEST_WAIT_TIME = 0.1  # Time to wait before retrying a request

    def __init__(
        self,
        model_id: OpenAIEmbeddingModel,
        alias: str | None = None,
        n_tokenizer_threads: int | None = None,
        max_requests_per_min: int = int(10_000 * 0.95),
        max_token_per_min: int = int(10_000_000 * 0.95),
        max_concurrent_requests: int = 10_000,
    ) -> None:
        super().__init__(model_id, alias, n_tokenizer_threads)

        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.rate_limiter = RateLimiter(max_requests_per_min, max_token_per_min)

    def embed(
        self, prompt_ids: list[str], prompts: list[str], truncate_dim: int | None = None
    ) -> tuple[list[str], FloatArray]:
        self.truncate_dim = truncate_dim
        batches = self._get_tokenized_batches(prompts)
        logger.info("Converted {} prompts into {} batch requests.", len(prompts), len(batches))

        result = self._event_loop.run_until_complete(self._process_batches(batches))

        # NOTE: We rely on the fact that the on-demand API returns results in the same order as the
        # input prompts, so we can safely return the prompt IDs in the same order.
        embeddings = np.vstack(list(chain.from_iterable(result)), dtype=np.float32)
        logger.debug("Stacked results into array of shape {}.", embeddings.shape)

        if len(embeddings) != len(prompt_ids):
            raise ValueError(
                f"Expected {len(prompt_ids)} embeddings, but got {len(embeddings)}. "
                "This may indicate an issue with the OpenAI API response."
            )

        return prompt_ids, embeddings

    async def _process_batches(self, batches: list[list[list[int]]]) -> list[list[list[float]]]:
        """Embed a list of batches asynchronously."""
        tasks: set[Task[list[list[float]]]] = set()
        with tqdm(
            desc="Embedding prompts",
            total=len(batches),
            leave=False,
            unit="batch",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            for batch in batches:
                task = asyncio.create_task(self._process_batch(batch))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                task.add_done_callback(lambda _: pbar.update(1))

            task_monitor = asyncio.create_task(self._monitor_tasks(tasks))

            results = await asyncio.gather(*tasks)

            task_monitor.cancel()
        return results

    async def _process_batch(self, prompts: list[list[int]]) -> list[list[float]]:
        """Submit a batch of tokenized prompts to the OpenAI API and return the results."""
        async with self.semaphore:
            # Wait until we can proceed with the next request
            await self.rate_limiter.wait(sum(len(p) for p in prompts))

            logger.debug("Submitting request with {} prompts.", len(prompts))
            kwargs: dict[str, Any] = {"input": prompts, "model": self.model_id}
            if self.truncate_dim is not None:
                kwargs["dimensions"] = self.truncate_dim
            response = await self._api_request(self.client.embeddings.create, **kwargs)
            logger.debug(
                "OpenAI generated {} tokens with model {}.",
                response.usage.total_tokens,
                response.model,
            )

            return [d.embedding for d in response.data]

    async def _monitor_tasks(self, tasks: set[Task[list[list[float]]]]) -> None:
        """Monitor background tasks and log their status."""
        while len(tasks) > 0:
            await asyncio.sleep(300)
            logger.debug("Number of remaining batches: {}", len(tasks))


class BatchOpenAIModel(AsyncOpenAIBase):
    """OpenAI model using the batch API.

    # See https://platform.openai.com/docs/guides/batch
    """

    # See https://platform.openai.com/docs/guides/batch#rate-limits for batch API rate limits
    BATCH_SIZE_LIMIT_MB = 200
    MAX_REQUESTS_PER_BATCH = 50_000
    MAX_CUSTOM_ID_LENGTH = 512  # OpenAI custom ID character limit

    def __init__(
        self,
        model_id: OpenAIEmbeddingModel,
        alias: str | None = None,
        n_tokenizer_threads: int | None = None,
        max_tokens_in_queue: int = int(4_000_000_000 * 0.95),
        max_tokens_per_batch: int = 25_000_000,
        batch_poll_timeout: int = 10,
    ) -> None:
        super().__init__(model_id, alias, n_tokenizer_threads)

        # Queue variables
        self.max_tokens_in_queue = max_tokens_in_queue
        self.max_requests_in_queue = float("inf")  # There currently is no request limit
        self.n_tokens_in_queue = 0
        self.n_requests_in_queue = 0
        self._queue_condition = asyncio.Condition()

        # Batch variables
        # We limit the number of tokens in a batch to avoid exceeding the 200MB batch size limit
        self.max_tokens_per_batch = max_tokens_per_batch
        self.batch_poll_timeout = batch_poll_timeout

    def embed(
        self, prompt_ids: list[str], prompts: list[str], truncate_dim: int | None = None
    ) -> tuple[list[str], FloatArray]:
        self.truncate_dim = truncate_dim
        jsonl_batches, id_mapping = self._convert_to_jsonl(prompts, prompt_ids)
        logger.info(
            "Converted {} prompts into {} jsonl batches.", len(prompts), len(jsonl_batches)
        )

        return self._event_loop.run_until_complete(
            self._process_batches(jsonl_batches, id_mapping)
        )

    def _convert_to_jsonl(
        self, prompts: list[str], prompt_ids: list[str]
    ) -> tuple[list[tuple[bytes, int, int]], dict[str, str]]:
        """Convert a list of prompts into a list of jsonl batches.

        Args:
            prompts: A list of prompts to be embedded.
            prompt_ids: A list of identifiers for each prompt. Since processing can be
                async or out-of-order, each prompt must have a unique ID.

        Returns:
            - A list of tuples containing a jsonl string in bytes as well as the token and request
                count per batch.
            - A mapping to prompt IDs to surrogate IDs if the prompt ID is longer than 512 chars.
        """
        tokenized_prompts = self.tokenize(prompts, max_length=self.max_seq_length)

        batches: list[tuple[bytes, int, int]] = []
        current_batch: list[dict[str, Any]] = []
        id_mapping: dict[str, str] = {}
        n_tokens = 0
        n_requests = 0
        for id_, tokens in zip(prompt_ids, tokenized_prompts, strict=True):
            if len(id_) > self.MAX_CUSTOM_ID_LENGTH:
                task_id = str(uuid.uuid4())
                id_mapping[id_] = task_id
            else:
                task_id = str(id_)
            task: dict[str, Any] = {
                "custom_id": task_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model_id, "input": tokens},
            }
            if self.truncate_dim is not None:
                task["body"]["dimensions"] = self.truncate_dim

            if (
                n_tokens + len(tokens) > self.max_tokens_per_batch
                or n_requests + 1 > self.MAX_REQUESTS_PER_BATCH
            ):
                # Convert the current batch into jsonl bytes and append it to batches
                jsonl_bytes = b"\n".join(orjson.dumps(task) for task in current_batch)
                batches.append((jsonl_bytes, n_tokens, n_requests))

                current_batch = [task]
                n_tokens = len(tokens)
                n_requests = 1
            else:
                # Add the task to the current batch
                current_batch.append(task)
                n_tokens += len(tokens)
                n_requests += 1

        # Add the last batch if it is not empty
        if current_batch:
            jsonl_bytes = b"\n".join(orjson.dumps(task) for task in current_batch)
            batches.append((jsonl_bytes, n_tokens, n_requests))

        logger.debug(
            "Created {} batches with {} tokens from prompts.",
            len(batches),
            [batch[1] for batch in batches],
        )
        return batches, id_mapping

    async def _process_batches(
        self, batches: list[tuple[bytes, int, int]], id_mapping: dict[str, str]
    ) -> tuple[list[str], FloatArray]:
        """Create a queue that submits batches to the OpenAI API and monitors their status.

        Args:
            batches: A list of tuples containing a jsonl payload and the batch's token and request
                count.
            id_mapping: A mapping from prompt IDs to surrogate IDs for too long prompts IDs.

        Returns:
            A tuple containing a list of prompt IDs and an array with their embeddings.
            The order is not guaranteed to match the input order.
        """
        queue: Queue[tuple[bytes, int, int] | None] = Queue()

        for batch in batches:
            await queue.put(batch)

        # Add sentinel to signal shutdown
        await queue.put(None)
        logger.debug("Filled the queue with {} batches.", len(batches))

        # Process the queue
        result = await self._consume_queue(queue, id_mapping)
        await queue.join()

        logger.info("Queue completed. Received {} prompt embeddings.", len(result[0]))
        return result

    async def _consume_queue(
        self, queue: Queue[tuple[bytes, int, int] | None], id_mapping: dict[str, str]
    ) -> tuple[list[str], FloatArray]:
        result: tuple[list[str], list[list[float]]] = ([], [])
        background_tasks: set[asyncio.Task[None]] = set()
        background_monitor = asyncio.create_task(self._monitor_tasks(queue, background_tasks))

        with tqdm(
            desc="Processing batches",
            total=queue.qsize() - 1,  # Exclude the sentinel
            leave=False,
            unit="batch",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            while True:
                item = await queue.get()
                if item is None:
                    # Stop the queue if the sentinel is received
                    queue.task_done()
                    break

                jsonl_bytes, n_tokens, n_requests = item

                # Bind the current token_count using a default parameter
                def predicate(tokens: int = n_tokens, requests: int = n_requests) -> bool:
                    return (self.n_tokens_in_queue + tokens <= self.max_tokens_in_queue) and (
                        self.n_requests_in_queue + requests <= self.max_requests_in_queue
                    )

                async with self._queue_condition:
                    await self._queue_condition.wait_for(predicate)
                    self.n_tokens_in_queue += n_tokens
                    self.n_requests_in_queue += n_requests

                batch = await self._submit_batch(jsonl_bytes)
                if batch:
                    task = asyncio.create_task(
                        self._monitor_batch(batch, n_tokens, n_requests, result, id_mapping),
                        name=batch.id,
                    )
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                    task.add_done_callback(lambda _: pbar.update(1))

                queue.task_done()

            if background_tasks:
                logger.info(
                    "Consumed all items in queue, waiting for {} background tasks to complete: {}",
                    len(background_tasks),
                    [task.get_name() for task in background_tasks],
                )
                await asyncio.gather(*background_tasks)

        background_monitor.cancel()
        return result[0], np.vstack(result[1], dtype=np.float32)

    async def _monitor_tasks(self, queue: Queue[Any], tasks: set[Task[None]]) -> None:
        while True:
            if len(tasks) == 0 and queue.empty():
                break
            await asyncio.sleep(300)
            logger.debug(
                "Background tasks remaining: {}, queue size: {}", len(tasks), queue.qsize()
            )

    async def _submit_batch(self, jsonl: bytes) -> Batch | None:
        batch_size_mb = len(jsonl) / 1024**2
        if batch_size_mb > self.BATCH_SIZE_LIMIT_MB:
            logger.error(
                "Batch file size exceeds {} MB limit. Skipping batch.", self.BATCH_SIZE_LIMIT_MB
            )
            return None
        logger.debug("Submitting batch of size {:.5f} MB.", batch_size_mb)

        input_file = await self._api_request(self.client.files.create, file=jsonl, purpose="batch")
        return await self._api_request(
            self.client.batches.create,
            input_file_id=input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

    async def _monitor_batch(
        self,
        batch: Batch,
        n_tokens: int,
        n_requests: int,
        result: tuple[list[str], list[list[float]]],
        id_mapping: dict[str, str],
    ) -> None:
        while True:
            await asyncio.sleep(self.batch_poll_timeout)
            try:
                batch_update = await self._api_request(self.client.batches.retrieve, batch.id)
            except retry_exceptions as e:
                logger.error("Exception while monitoring batch {}: {}", batch.id, e)
                continue

            match batch_update.status:
                case "failed":
                    logger.error("Batch {} failed.", batch_update.id)
                    await self._process_errors(batch_update, True)
                    break
                case "expired" | "cancelling" | "cancelled":
                    logger.error("Batch {} expired or was cancelled.", batch_update.id)
                    break
                case "completed":
                    logger.debug("Batch {} completed. Processing results...", batch_update.id)
                    await self._process_results(batch_update, result, id_mapping)
                    break

        async with self._queue_condition:
            self.n_tokens_in_queue -= n_tokens
            self.n_requests_in_queue -= n_requests
            self._queue_condition.notify_all()

    async def _process_results(
        self, batch: Batch, result: tuple[list[str], list[list[float]]], id_mapping: dict[str, str]
    ) -> None:
        """Process the results of a completed batch and append them to the result dictionary.

        Args:
            batch: The completed batch object containing the results.
            result: The dictionary to append the results to.
            id_mapping: A mapping from prompt IDs to surrogate IDs for too long prompt IDs.
        """
        if batch.output_file_id is None:
            logger.error("Batch {} completed but has no output file.", batch.id)
            return

        batch_result = await self._api_request(self.client.files.content, batch.output_file_id)

        n_lines = 0
        async for line in batch_result.response.aiter_lines():
            try:
                data = orjson.loads(line)
                prompt_id = cast("str", id_mapping.get(data["custom_id"], data["custom_id"]))
                result[0].append(prompt_id)
                result[1].append(data["response"]["body"]["data"][0]["embedding"])
                n_lines += 1
            except orjson.JSONDecodeError as e:
                logger.warning("Skipping invalid line in batch {}: {}", batch.id, e)
                continue
        logger.debug("Processed {} results from batch {}.", n_lines, batch.id)

        await self._process_errors(batch, False)

    async def _process_errors(self, batch: Batch, failed: bool) -> None:
        if batch.error_file_id is None:
            if failed:
                logger.error("Batch {} failed but has no error file: {}", batch.id, batch)
            return

        errors = await self._api_request(self.client.files.content, batch.error_file_id)
        logger.error("Error file for batch {}:\n{}", batch.id, errors.text)
