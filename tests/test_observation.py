"""观察注入与模式规则的静态校验。

live_adapter 的依赖（aiohttp/openai/memory_summarizer）在轻量环境里不一定装得齐，
这里不 import 模块，直接读源码做断言：规则都在两个 prompt 常量里，注入逻辑只有
一处 startswith("[") 分支。
"""

from pathlib import Path

import pytest

ADAPTER = (
    Path(__file__).resolve().parents[1] / "services" / "webinfer" / "live_adapter.py"
)


@pytest.fixture(scope="module")
def source() -> str:
    return ADAPTER.read_text(encoding="utf-8")


def _extract_prompt(source: str, name: str) -> str:
    marker = f'{name} = """'
    start = source.index(marker) + len(marker)
    end = source.index('"""', start)
    return source[start:end]


def test_culinary_rules_in_zh_prompt(source: str) -> None:
    prompt = _extract_prompt(source, "DEFAULT_SYSTEM_PROMPT_ZH")
    assert "膳食模式" in prompt
    assert "[膳食观察]" in prompt
    assert "[膳食事件]" in prompt
    # 关键约束：事件必须开口、数字只能照抄、不重复到点、不凭外观判断肉类。
    assert "必须开口回应" in prompt
    assert "只能照抄" in prompt
    assert "不要重复" in prompt
    assert "不要凭外观判断肉类熟透" in prompt
    # 菜谱由系统提供：未拿到 [膳食观察] 前不自编配料；用量问题据此直答。
    assert "菜谱由系统提供" in prompt
    assert "本菜用量" in prompt
    assert "不要增减、替换或换算成别的单位" in prompt


def test_culinary_rules_in_en_prompt(source: str) -> None:
    prompt = _extract_prompt(source, "DEFAULT_SYSTEM_PROMPT_EN")
    assert "Culinary mode" in prompt
    assert '"[膳食观察]"' in prompt
    assert '"[膳食事件]"' in prompt
    assert "requires a spoken reply" in prompt
    assert "verbatim" in prompt
    assert "appearance alone" in prompt
    assert "the system provides the recipe" in prompt
    assert "本菜用量" in prompt
    assert "never add, drop, substitute, or convert units" in prompt


def test_tagged_observation_passes_through_verbatim(source: str) -> None:
    """[膳食观察] 与 [空间观察] 一样带前缀 → 走原样注入分支，不会被再加 [动作观察]。"""
    assert 'observation_text.startswith("[")' in source
    assert '"[动作观察]\\n" + observation_text' in source


def test_culinary_observation_prefix_is_tagged() -> None:
    """[膳食观察] 前缀触发 startswith("[") 分支——与 adapter 的判断方式一致。"""
    assert "[膳食观察]".startswith("[")
    assert "[膳食事件]".startswith("[")
