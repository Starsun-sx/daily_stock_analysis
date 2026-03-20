# -*- coding: utf-8 -*-
"""
Cherry Studio / lyngpt 风格 API 兼容层。

当第三方 API 使用 /v1/responses 与 input/output 结构（非 OpenAI /chat/completions）
时，通过 OPENAI_CHERRY_COMPAT=true 启用本适配器，将 OpenAI 风格 messages 转为
Cherry 请求并解析响应为统一文本。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Cherry 要求的部分请求头（与 Cherry Studio 一致）
CHERRY_HEADERS = {
    "Accept": "application/json",
    "HTTP-Referer": "https://cherry-ai.com",
    "X-Title": "Cherry Studio",
}


def _message_content_to_text(content: Any) -> str:
    """从单条 message 的 content 提取纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content).strip()


def _messages_to_single_input_text(messages: List[Dict[str, Any]]) -> str:
    """将 OpenAI 风格 messages 合并为单段纯文本，供 Cherry input 字符串格式使用。"""
    system_parts: List[str] = []
    user_text: Optional[str] = None
    for msg in messages:
        role = (msg.get("role") or "user").strip().lower()
        text = _message_content_to_text(msg.get("content"))
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            user_text = text
            break
        elif role == "assistant":
            system_parts.append(f"[Assistant]\n{text}")
    if user_text is None:
        user_text = ""
    full_text = "\n\n".join(system_parts).strip()
    if full_text:
        full_text = full_text + "\n\n[User]\n" + user_text
    else:
        full_text = user_text
    return full_text


def _build_responses_url(base_url: str) -> str:
    """根据 BASE 得到 /v1/responses 或 /responses 的完整 URL。"""
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def call_cherry_api(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: int = 120,
) -> Tuple[str, str, Dict[str, Any]]:
    """调用 Cherry/lyngpt 风格 API，返回与 _call_litellm 一致的 (content, model, usage)。

    Args:
        base_url: 如 https://lyngpt.lynxi.com/api 或 https://lyngpt.lynxi.com/api/v1
        api_key: Bearer 使用的 Key
        model: 模型名，如 gpt-5.2（无需 openai/ 前缀）
        messages: OpenAI 风格 [{"role":"system","content":"..."},{"role":"user","content":"..."}]
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        timeout: 请求超时秒数

    Returns:
        (response_text, model_used, usage_dict)
        usage_dict 含 prompt_tokens, completion_tokens, total_tokens（若 API 返回了 usage）

    Raises:
        ValueError: 参数无效或响应无文本
        requests.RequestException: 网络/HTTP 错误
    """
    base_url = (base_url or "").strip()
    if not base_url or not api_key or not model:
        raise ValueError("Cherry API 需要 base_url、api_key 和 model")

    url = _build_responses_url(base_url)
    model_short = model.split("/")[-1] if "/" in model else model
    input_text = _messages_to_single_input_text(messages)
    if not input_text:
        raise ValueError("messages 转换后为空")

    # 与 test_api2 一致：input 为纯字符串
    payload: Dict[str, Any] = {
        "model": model_short,
        "input": input_text,
        "temperature": temperature,
    }
    if max_tokens > 0:
        payload["max_output_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **CHERRY_HEADERS,
    }

    logger.debug("Cherry API request url=%s model=%s", url, model_short)
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # 从 output[].content[].text 抽取文本（兼容 output[0]["content"][0]["text"] 简式）
    text_parts: List[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text", "")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
    response_text = "\n".join(text_parts).strip()
    if not response_text:
        raise ValueError("Cherry API 响应中无 output 文本内容")

    usage: Dict[str, Any] = {}
    if "usage" in data and isinstance(data["usage"], dict):
        u = data["usage"]
        usage = {
            "prompt_tokens": u.get("input_tokens") or u.get("prompt_tokens", 0),
            "completion_tokens": u.get("output_tokens") or u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }

    return (response_text, model, usage)
