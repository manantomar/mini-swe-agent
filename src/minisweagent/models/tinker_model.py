"""Tinker-based model class for mini-swe-agent.

Uses Tinker's SamplingClient for inference, handling chat template
rendering and tool call parsing locally.

Usage:
    Set TINKER_API_KEY and either:
    1. Use a base model: --model-class tinker -m Qwen/Qwen3-8B
    2. Use a trained checkpoint: -c "model.tinker_checkpoint_path=tinker://..."
"""

import json
import logging
import os
import re
import time
import uuid
from typing import Any, Literal

import tinker
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("tinker_model")

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Bash command to run"}},
                "required": ["command"],
            },
        },
    }
]


class TinkerModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-8B"
    """Base model name (HuggingFace ID)."""
    tinker_checkpoint_path: str = ""
    """Tinker checkpoint path to load trained weights from (e.g., tinker://...).
    If empty, uses the base model without fine-tuning."""
    model_kwargs: dict[str, Any] = {}
    """Additional kwargs (temperature, max_tokens, etc.)."""
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""
    max_tokens: int = 2048


class TinkerModel:
    abort_exceptions: list[type[Exception]] = [KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = TinkerModelConfig(**kwargs)
        self._setup_clients()

    def _setup_clients(self):
        """Initialize Tinker service, sampling client, and tokenizer."""
        logger.info(f"Connecting to Tinker (model={self.config.model_name})...")
        self._service_client = tinker.ServiceClient()

        if self.config.tinker_checkpoint_path:
            path = self.config.tinker_checkpoint_path
            logger.info(f"Loading from checkpoint: {path}")
            if "/sampler_weights/" in path:
                # Sampler-only path — lightweight, no training session
                self._sampling_client = self._service_client.create_sampling_client(model_path=path)
            else:
                # Training checkpoint — creates a training session (expensive, avoid in parallel)
                tc = self._service_client.create_training_client_from_state(path)
                self._sampling_client = tc.save_weights_and_get_sampling_client()
            self._tokenizer = self._sampling_client.get_tokenizer()
        else:
            logger.info(f"Using base model: {self.config.model_name}")
            self._sampling_client = self._service_client.create_sampling_client(
                base_model=self.config.model_name
            )
            self._tokenizer = self._sampling_client.get_tokenizer()

        # Determine stop sequences from the chat template
        self._stop_sequences = ["<|im_end|>"]
        logger.info("Tinker model ready.")

    def _messages_to_tokens(self, messages: list[dict]) -> list[int]:
        """Convert chat messages to token IDs using the model's chat template."""
        cleaned = []
        for m in messages:
            msg = {"role": m["role"]}
            if m.get("content"):
                msg["content"] = m["content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            cleaned.append(msg)

        text = self._tokenizer.apply_chat_template(
            cleaned, tools=TOOLS_SPEC, tokenize=False, add_generation_prompt=True
        )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _parse_tool_calls_from_text(self, text: str) -> list[dict]:
        """Parse <tool_call>...</tool_call> blocks from generated text."""
        tool_calls = []
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        for match in matches:
            try:
                tc_data = json.loads(match)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tc_data.get("name", ""),
                        "arguments": json.dumps(tc_data.get("arguments", {})),
                    },
                })
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _query(self, messages: list[dict], **kwargs) -> dict:
        """Generate a response using Tinker's sampling client."""
        prompt_tokens = self._messages_to_tokens(messages)
        prompt = tinker.types.ModelInput.from_ints(prompt_tokens)

        temperature = self.config.model_kwargs.get("temperature", 0.0)
        temperature = kwargs.get("temperature", temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            stop=self._stop_sequences,
        )

        future = self._sampling_client.sample(
            prompt=prompt, num_samples=1, sampling_params=params
        )
        result = future.result()
        sequence = result.sequences[0]

        # Extract raw token ids and logprobs before any text processing
        raw_token_ids = list(sequence.tokens)
        raw_logprobs = list(sequence.logprobs)

        response_text = self._tokenizer.decode(sequence.tokens, skip_special_tokens=False)
        # Strip trailing special tokens
        for stop in self._stop_sequences:
            response_text = response_text.split(stop)[0]

        # Parse tool calls
        tool_calls = self._parse_tool_calls_from_text(response_text)

        # Clean content (remove tool_call tags)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()

        message = {
            "role": "assistant",
            "content": content or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Build response in OpenAI-like format
        return {
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(raw_token_ids),
                "total_tokens": len(prompt_tokens) + len(raw_token_ids),
            },
            "logprob_data": {
                "token_ids": raw_token_ids,
                "logprobs": raw_logprobs,
                "sum_logprob": sum(raw_logprobs),
                "n_tokens": len(raw_logprobs),
            },
        }

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def query(self, messages: list[dict], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        GLOBAL_MODEL_STATS.add(0.0)  # no cost for Tinker inference
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            "cost": 0.0,
            "timestamp": time.time(),
            "logprobs": response.get("logprob_data", {}),
        }
        return message

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse tool calls from the response."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class _DictToObj:
    """Simple wrapper to convert dict to object with attribute access."""

    def __init__(self, d: dict):
        self._d = d
        self.id = d.get("id")
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        self.arguments = d.get("arguments")
