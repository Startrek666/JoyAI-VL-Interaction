# -*- coding: utf-8 -*-
"""运动观察通道（[动作观察]）单元测试。不连 vLLM，全部本地构造。"""

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live_adapter
from live_adapter import (
    AdapterConfig,
    SessionState,
    StreamingInferAdapter,
    _extract_observation,
    _next_observation,
)


@pytest.fixture()
def adapter(tmp_path):
    config = AdapterConfig(
        enable_summarizer=False,
        frame_save_dir=str(tmp_path / "frames"),
        language="zh",
    )
    return StreamingInferAdapter(config)


# ---------- _extract_observation ----------

def test_extract_observation_top_level():
    assert _extract_observation({"observation": "深蹲 膝角 95°"}) == "深蹲 膝角 95°"


def test_extract_observation_from_extra_body_dict():
    payload = {"extra_body": {"observation": "平板支撑 30秒"}}
    assert _extract_observation(payload) == "平板支撑 30秒"


def test_extract_observation_missing_and_non_string():
    assert _extract_observation({}) == ""
    assert _extract_observation({"observation": 123}) == ""
    assert _extract_observation({"observation": None}) == ""
    assert _extract_observation({"extra_body": "not-a-dict"}) == ""


def test_extract_observation_strips_and_truncates_800():
    raw = "  " + "x" * 900 + "  "
    out = _extract_observation({"observation": raw})
    assert out == "x" * 800


def test_extract_observation_top_level_wins_over_extra_body():
    payload = {"observation": "顶层", "extra_body": {"observation": "extra"}}
    assert _extract_observation(payload) == "顶层"


# ---------- _next_observation ----------

def test_next_observation_dedup_and_clear():
    state = SessionState(session_id="t")
    assert _next_observation(state, "obs A") == "obs A"
    assert state.last_observation == "obs A"
    # 内容相同：不重复追加
    assert _next_observation(state, "obs A") is None
    assert _next_observation(state, "obs B") == "obs B"
    # 空观察复位去重记录，恢复后能再次注入同一文本
    assert _next_observation(state, "") is None
    assert state.last_observation is None
    assert _next_observation(state, "obs B") == "obs B"


# ---------- _build_internal_user_message ----------

def _texts(message):
    return [item for item in message["content"] if item.get("type") == "text"]


def test_build_message_appends_observation_after_images(adapter):
    message = adapter._build_internal_user_message(
        time_ranges=["1.0 seconds"],
        image_paths=["/a.jpg", "/b.jpg"],
        query_text="帮我看看",
        observation_text="动作：深蹲，次数：3",
    )
    content = message["content"]
    # 顺序：用户问题 → 时间标记 → 图片 → [动作观察]
    assert content[-1] == {"type": "text", "text": "[动作观察]\n动作：深蹲，次数：3"}
    image_indexes = [i for i, item in enumerate(content) if item.get("type") == "image"]
    assert image_indexes == [2, 3]
    assert "帮我看看" in content[0]["text"]


def test_build_message_without_observation_unchanged(adapter):
    kwargs = dict(time_ranges=["1.0 seconds"], image_paths=["/a.jpg"], query_text="帮我看看")
    plain = adapter._build_internal_user_message(**kwargs)
    with_none = adapter._build_internal_user_message(**kwargs, observation_text=None)
    with_empty = adapter._build_internal_user_message(**kwargs, observation_text="")
    assert plain == with_none == with_empty
    assert all("[动作观察]" not in item.get("text", "") for item in _texts(plain))


def test_internal_message_to_openai_converts_observation(adapter):
    message = adapter._build_internal_user_message(
        image_paths=[], observation_text="次数：5"
    )
    converted = live_adapter._internal_message_to_openai(message)
    assert converted["content"][-1] == {"type": "text", "text": "[动作观察]\n次数：5"}


# ---------- _handle_chat_payload（强制沉默路径，不调模型） ----------

def _frame_payload(observation=None):
    data_url = "data:image/jpeg;base64," + base64.b64encode(b"fake-jpeg").decode("ascii")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url}}],
            }
        ]
    }
    if observation is not None:
        payload["observation"] = observation
    return payload


def _mock_request():
    from aiohttp.test_utils import make_mocked_request

    return make_mocked_request(
        "POST", "/v1/chat/completions", headers={"x-streaming-session": "t"}
    )


def _last_user_message(state) -> dict:
    """chunk 里最后一条 user 消息（assistant 回复是纯字符串 content，要跳过）。"""
    for message in reversed(state.current_chunk["messages"]):
        if message.get("role") == "user" and isinstance(message.get("content"), list):
            return message
    return {}


def _has_observation_segment(message) -> bool:
    return any(
        item.get("type") == "text" and str(item.get("text") or "").startswith("[动作观察]")
        for item in message.get("content") or []
    )


@pytest.mark.asyncio
async def test_chat_payload_appends_dedups_and_reinjects(adapter):
    state = SessionState(session_id="t")
    request = _mock_request()

    await adapter._handle_chat_payload(state, _frame_payload("深蹲 膝角 95°"), request)
    first = _last_user_message(state)
    assert _has_observation_segment(first)
    assert first["content"][-1]["text"] == "[动作观察]\n深蹲 膝角 95°"
    assert state.predictions[-1]["input"]["observation"] == "深蹲 膝角 95°"
    # 观察不进入用户问题通道
    assert state.current_query_text is None

    # 相同观察：不重复追加
    await adapter._handle_chat_payload(state, _frame_payload("深蹲 膝角 95°"), request)
    second = _last_user_message(state)
    assert not _has_observation_segment(second)
    assert state.predictions[-1]["input"]["observation"] is None

    # 清空后：同一文本能再次注入
    await adapter._handle_chat_payload(state, _frame_payload(), request)
    await adapter._handle_chat_payload(state, _frame_payload("深蹲 膝角 95°"), request)
    fourth = _last_user_message(state)
    assert _has_observation_segment(fourth)


def test_default_prompt_zh_contains_motion_rules():
    assert "## 运动模式" in live_adapter.DEFAULT_SYSTEM_PROMPT_ZH
    assert "[动作观察]" in live_adapter.DEFAULT_SYSTEM_PROMPT_ZH
