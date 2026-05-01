"""vLLM-based model class for mini-swe-agent.

Connects to a local vLLM server (OpenAI-compatible API) for fast inference.
vLLM handles chat templates and tool-call parsing natively (hermes parser).

Usage:
    1. Start vLLM:  bash rl/serve_vllm.sh --model Qwen/Qwen3-8B
    2. Run agent:   mini-extra swebench ... --model-class vllm -m Qwen/Qwen3-8B
"""

import logging
import time
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("vllm_model")


class VllmModelConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-8B"
    """Model name as registered in the vLLM server."""
    api_base: str = "http://localhost:8234/v1"
    """Base URL of the vLLM OpenAI-compatible API."""
    model_kwargs: dict[str, Any] = {}
    """Additional kwargs passed to the API (temperature, max_tokens, etc.)."""
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"
    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class VllmModel:
    abort_exceptions: list[type[Exception]] = [KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = VllmModelConfig(**kwargs)
        self._client = OpenAI(base_url=self.config.api_base, api_key="not-needed")
        logger.info(f"vLLM model: {self.config.model_name} at {self.config.api_base}")

    # kwargs that are litellm-specific and not supported by the openai client
    _UNSUPPORTED_KWARGS = {"drop_params", "set_cache_control", "parallel_tool_calls"}

    def _query(self, messages: list[dict], **kwargs):
        merged = {k: v for k, v in (self.config.model_kwargs | kwargs).items() if k not in self._UNSUPPORTED_KWARGS}
        return self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            tools=[BASH_TOOL],
            logprobs=True,
            top_logprobs=1,
            **merged,
        )

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def query(self, messages: list[dict], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        GLOBAL_MODEL_STATS.add(0.0)
        message = response.choices[0].message.model_dump()

        # Extract per-token logprobs from response
        logprob_data = self._extract_logprobs(response)

        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            "cost": 0.0,
            "timestamp": time.time(),
            "logprobs": logprob_data,
        }
        return message

    def _extract_logprobs(self, response) -> dict:
        """Extract per-token logprobs from the vLLM response."""
        choice = response.choices[0]
        if not choice.logprobs or not choice.logprobs.content:
            return {"token_ids": [], "logprobs": [], "tokens": []}
        token_ids = []
        logprobs = []
        tokens = []
        for token_logprob in choice.logprobs.content:
            tokens.append(token_logprob.token)
            logprobs.append(token_logprob.logprob)
            # top_logprobs[0] has the token info including id
            if token_logprob.top_logprobs:
                token_ids.append(getattr(token_logprob.top_logprobs[0], "token_id", 0) or 0)
            else:
                token_ids.append(0)
        return {
            "token_ids": token_ids,
            "logprobs": logprobs,
            "tokens": tokens,
            "sum_logprob": sum(logprobs),
            "n_tokens": len(logprobs),
        }

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
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
