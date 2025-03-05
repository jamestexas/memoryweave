import asyncio
import logging
from collections.abc import AsyncGenerator

from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming generation for different model architectures."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.model = llm_provider.model
        self.tokenizer = llm_provider.tokenizer
        self.device = llm_provider.device
        self._check_capabilities()

    def _check_capabilities(self):
        """Check model streaming capabilities."""
        self.has_native_streaming = hasattr(self.model.generate, "streamer") or hasattr(
            self.model, "stream"
        )

    async def stream(
        self, prompt: str, max_new_tokens: int = 512, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if self.has_native_streaming:
            # Use native streaming if available
            async for token in self._native_stream(inputs, max_new_tokens, **kwargs):
                yield token
        else:
            # Fall back to simulated streaming
            async for token in self._simulated_stream(inputs, max_new_tokens, **kwargs):
                yield token

    async def _native_stream(self, inputs, max_new_tokens, **kwargs):
        """Use the model's native streaming capabilities."""
        try:
            # Try to use HuggingFace TextStreamer if available
            from transformers import TextStreamer

            streamer = TextStreamer(self.tokenizer)
            token_queue = asyncio.Queue()

            async def _stream_tokens():
                try:
                    async for token in streamer:
                        await token_queue.put(token)
                    await token_queue.put(None)  # Signal end of generation
                except Exception as e:
                    logger.error(f"Error in token streaming: {e}")
                    await token_queue.put(None)

            # Start streaming task
            asyncio.create_task(_stream_tokens())

            # Start generation
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
                "streamer": streamer,
            }

            self.model.generate(**generation_kwargs)

            # Yield tokens as they become available
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                yield token
        except (ImportError, Exception) as e:
            logger.error(f"Native streaming failed: {e}, falling back to simulated streaming")
            async for token in self._simulated_stream(inputs, max_new_tokens, **kwargs):
                yield token

    async def _simulated_stream(self, inputs, max_new_tokens, **kwargs):
        """Simulate streaming for models without native support."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.7),
                do_sample=kwargs.get("do_sample", True),
                top_p=kwargs.get("top_p", 0.9),
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Determine the start of the response
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response_text = full_text[len(prompt_text) :]

        # Simulate streaming by yielding chunks
        for _i, token in enumerate(response_text.split()):
            yield token + " "
            await asyncio.sleep(0.01)  # Simulate typing delay
