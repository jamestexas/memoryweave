import logging

import torch
from rich.logging import RichHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.benchmarks.utils.perf_timer import timer
from memoryweave.utils import _get_device

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


class LLMProvider:
    """Manages LLM loading and generation."""

    _global_model = None
    _global_tokenizer = None

    def __init__(self, model_name=DEFAULT_MODEL, device="auto", **model_kwargs):
        self.model_name = model_name
        self.device = _get_device(device)
        self.model_kwargs = model_kwargs

        if LLMProvider._global_model is None:
            self._load_model()
            LLMProvider._global_model = self.model
            LLMProvider._global_tokenizer = self.tokenizer
        else:
            self.model = LLMProvider._global_model
            self.tokenizer = LLMProvider._global_tokenizer

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading LLM: {self.model_name} - {self.device} - {self.model_kwargs}")

        # Configure torch dtype based on device
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        logging.debug("Setting tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logging.debug("Before instantiating LLM")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            **self.model_kwargs,
        )
        logging.debug("After instantiating LLM")

    @timer("llm_inference")
    def generate(self, prompt, max_new_tokens=512, **kwargs):
        """Generate a response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=kwargs.get("temperature", 0.7),
                do_sample=kwargs.get("do_sample", True),
                top_p=kwargs.get("top_p", 0.9),
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_response[len(prompt) :].strip()

        return response_text
