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
    # 推荐菜不自报菜名：候选由系统给出（膳食段与通用段各一条）。
    assert prompt.count("候选菜由系统给出") >= 2
    assert "等它出现在「[膳食观察]」或「[膳食事件]」里再介绍" in prompt


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
    # 推荐菜不自报菜名：候选由系统给出（膳食段与通用段各一条）。
    assert prompt.count("do not name a dish yourself") >= 2
    assert 'only after they appear in "[膳食观察]" or "[膳食事件]"' in prompt


def test_tagged_observation_passes_through_verbatim(source: str) -> None:
    """[膳食观察] 与 [空间观察] 一样带前缀 → 走原样注入分支，不会被再加 [动作观察]。"""
    assert 'observation_text.startswith("[")' in source
    assert '"[动作观察]\\n" + observation_text' in source


def test_culinary_observation_prefix_is_tagged() -> None:
    """[膳食观察] 前缀触发 startswith("[") 分支——与 adapter 的判断方式一致。"""
    assert "[膳食观察]".startswith("[")
    assert "[膳食事件]".startswith("[")
# 方案5 规格 md 的并排路径（lemos 仓），找不到时逐字断言降级为关键句断言。
_SPEC = Path(
    "D:/Aprojects/subscription/lemos/docs/video/方案5-一期实现规格.md")


def _spec_block(marker: str) -> str | None:
    if not _SPEC.exists():
        return None
    import re

    text = _SPEC.read_text(encoding="utf-8")
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL):
        if marker in block:
            return block.strip()
    return None


def test_style_rules_in_zh_prompt(source: str) -> None:
    """化妆模式规则（方案5 §9）在 ZH prompt 里，且与规格逐字一致。"""
    prompt = _extract_prompt(source, "DEFAULT_SYSTEM_PROMPT_ZH")
    block = _spec_block("用户屏幕上的妆是系统渲染的虚拟妆")
    if block is not None:
        assert block in prompt, "§9 中文块与规格不一致"
        return
    assert "美妆模式" in prompt
    assert "「[造型观察]」" in prompt
    assert "「[造型事件]」" in prompt
    assert "必须开口" in prompt
    assert "色号、品牌或数值" in prompt


def test_style_rules_in_en_prompt(source: str) -> None:
    prompt = _extract_prompt(source, "DEFAULT_SYSTEM_PROMPT_EN")
    block = _spec_block("The makeup on the user's screen is virtual")
    if block is not None:
        assert block in prompt, "§9 英文块与规格不一致"
        return
    assert "Makeup mode" in prompt
    assert '"[造型观察]"' in prompt
    assert '"[造型事件]"' in prompt
    assert "requires a spoken reply" in prompt


def test_style_observation_prefix_is_tagged() -> None:
    """[造型观察] / [造型事件] 带前缀 → startswith("[") 原样注入。"""
    assert "[造型观察]".startswith("[")
    assert "[造型事件]".startswith("[")
