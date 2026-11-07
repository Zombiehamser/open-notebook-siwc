"""Text-only SIWC provider for Open Notebook, using the public Responses API."""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass

import httpx
from esperanto import LanguageModel
from esperanto.common_types.response import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    Message,
    StreamChoice,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from open_notebook.ai.siwc_auth import (
    RESOURCE,
    PlanLimitReached,
    SignInRequired,
    SIWCConfigurationError,
    SIWCSession,
    check_response,
)
from open_notebook.exceptions import (
    ExternalServiceError,
    IncompleteGenerationError,
    NetworkError,
)


def request_input(messages):
    result = []
    roles = {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "developer",
        "developer": "developer",
    }
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else message.type
        content = (
            message.get("content", "") if isinstance(message, dict) else message.content
        )
        if role not in roles:
            raise SIWCConfigurationError(
                "This SIWC text prototype does not accept tool messages."
            )
        if isinstance(content, list):
            if any(
                not isinstance(block, dict) or block.get("type") != "text"
                for block in content
            ):
                raise SIWCConfigurationError(
                    "This SIWC prototype accepts extracted document text only."
                )
            content = "\n".join(block["text"] for block in content)
        if not isinstance(content, str):
            raise SIWCConfigurationError("Invalid SIWC text content.")
        result.append({"role": roles[role], "content": content})
    return result


def events(lines):
    """SSE frames may contain several data lines; transport chunks are irrelevant."""
    data: list[str] = []
    size = 0
    for line in lines:
        if not line:
            if data:
                payload = "\n".join(data)
                if payload != "[DONE]":
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        raise IncompleteGenerationError(
                            "Invalid SIWC stream event."
                        ) from None
                data, size = [], 0
        elif line.startswith("data:"):
            part = line[5:].lstrip(" ")
            size += len(part)
            if size > 4 * 1024 * 1024:
                raise IncompleteGenerationError(
                    "SIWC stream event exceeded the safety limit."
                )
            data.append(part)
    if data:
        raise IncompleteGenerationError(
            "SIWC stream ended with an incomplete SSE frame."
        )


class SIWCChatModel(BaseChatModel):
    model_name: str
    account_id: str
    json_output: bool = False

    @property
    def _llm_type(self):
        return "chatgpt_siwc"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        if stop or kwargs.get("tools"):
            raise SIWCConfigurationError(
                "SIWC text prototype does not support stop sequences or tool calls."
            )
        session = SIWCSession()
        payload = {
            "model": self.model_name,
            "input": request_input(messages),
            "store": False,
            "stream": True,
        }
        if self.json_output:
            payload["text"] = {"format": {"type": "json_object"}}
        completed = False
        text_seen = False
        received = 0
        # Bound each lane independently: one foreground and one background request.
        with session.lock(
            "inference-worker"
            if os.environ.get("OPEN_NOTEBOOK_PROCESS_ROLE") == "worker"
            else "inference"
        ):
            token = session.access_token(self.account_id)
            try:
                with (
                    session.http() as client,
                    client.stream(
                        "POST",
                        RESOURCE + "/responses",
                        headers={"Authorization": "Bearer " + token},
                        json=payload,
                    ) as response,
                ):
                    check_response(response)
                    for event in events(response.iter_lines()):
                        kind = event.get("type")
                        if kind == "response.output_text.delta":
                            delta = event.get("delta", "")
                            if not isinstance(delta, str):
                                raise IncompleteGenerationError(
                                    "Invalid SIWC text delta."
                                )
                            received += len(delta)
                            if received > 32 * 1024 * 1024:
                                raise IncompleteGenerationError(
                                    "SIWC text output exceeded the local safety limit."
                                )
                            if delta:
                                text_seen = True
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(content=delta)
                                )
                        elif kind in ("response.failed", "error"):
                            error = (
                                event.get("response", {}).get("error")
                                or event.get("error")
                                or event
                            )
                            code = error.get("code", "")
                            if code in (
                                "subscription_sharing_usage_limit_exceeded",
                                "rate_limit_exceeded",
                            ):
                                raise PlanLimitReached(
                                    "ChatGPT plan limit reached during generation. Review ChatGPT Settings → Usage."
                                )
                            if code == "subscription_sharing_usage_unavailable":
                                raise SignInRequired(
                                    "ChatGPT plan usage is unavailable for this connection."
                                )
                            raise ExternalServiceError(
                                "SIWC generation failed. No completed answer was received."
                            )
                        elif kind == "response.incomplete":
                            raise IncompleteGenerationError(
                                "ChatGPT returned an incomplete response."
                            )
                        elif kind == "response.completed":
                            if (
                                event.get("response", {}).get("status", "completed")
                                != "completed"
                            ):
                                raise IncompleteGenerationError(
                                    "SIWC did not confirm completed inference."
                                )
                            completed = True
                            break
            except httpx.HTTPError:
                raise NetworkError(
                    "SIWC stream failed through the configured proxy; no direct fallback was attempted."
                ) from None
        if not completed or not text_seen:
            raise IncompleteGenerationError(
                "SIWC stream ended without a completed, non-empty answer."
            )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata={"finish_reason": "stop"}
            )
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "".join(
            chunk.text for chunk in self._stream(messages, stop, run_manager, **kwargs)
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


@dataclass
class SIWCLanguageModel(LanguageModel):
    account_id: str = ""

    @property
    def provider(self):
        return "chatgpt_siwc"

    def _get_default_model(self):
        raise SIWCConfigurationError(
            "Choose a model from the signed-in ChatGPT account catalog."
        )

    def _get_models(self):
        return []  # The CLI uses live account-specific discovery, never a static catalog.

    def to_langchain(self):
        if not self.account_id or not self.model_name:
            raise SIWCConfigurationError(
                "SIWC requires a registered account and model."
            )
        return SIWCChatModel(
            model_name=self.model_name,
            account_id=self.account_id,
            json_output=bool(self.structured),
        )

    def chat_complete(
        self,
        messages,
        stream=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        validate_tool_calls=False,
        max_tokens=None,
        temperature=None,
        top_p=None,
    ):
        model = self.to_langchain()
        if tools or self.tools or tool_choice not in (None, "none", "auto"):
            raise SIWCConfigurationError(
                "Tool calls are outside the SIWC text prototype."
            )
        if stream if stream is not None else self.streaming:

            def chunks():
                completion_id, created = "siwc-" + uuid.uuid4().hex, int(time.time())
                for chunk in model.stream(messages):
                    if not chunk.content and not chunk.response_metadata.get(
                        "finish_reason"
                    ):
                        continue
                    yield ChatCompletionChunk(
                        id=completion_id,
                        model=self.model_name,
                        created=created,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    role="assistant", content=chunk.content
                                ),
                                finish_reason=chunk.response_metadata.get(
                                    "finish_reason"
                                ),
                            )
                        ],
                    )

            return chunks()
        response = model.invoke(messages)
        return ChatCompletion(
            id="siwc-" + uuid.uuid4().hex,
            model=self.model_name,
            provider=self.provider,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response.content),
                    finish_reason="stop",
                )
            ],
        )

    async def achat_complete(
        self,
        messages,
        stream=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        validate_tool_calls=False,
        max_tokens=None,
        temperature=None,
        top_p=None,
    ):
        result = await asyncio.to_thread(
            self.chat_complete,
            messages,
            stream,
            tools,
            tool_choice,
            parallel_tool_calls,
            validate_tool_calls,
            max_tokens,
            temperature,
            top_p,
        )
        if not (stream if stream is not None else self.streaming):
            return result

        async def chunks():
            sentinel = object()
            try:
                while True:
                    value = await asyncio.to_thread(next, result, sentinel)
                    if value is sentinel:
                        break
                    yield value
            finally:
                await asyncio.to_thread(result.close)

        return chunks()
