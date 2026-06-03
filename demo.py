"""
toolsclaw 调用 demo

核心 API:
    from toolsclaw import ToolsClaw

    agent = ToolsClaw.from_config(workspace="./my-project", skills_dir="./my-skills")
    result = await agent.run("Hello")
"""

import asyncio
from pathlib import Path
from typing import Any

from toolsclaw import ToolsClaw, AgentHook
from toolsclaw.hook import AgentHookContext

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

API_KEY = "tp-cs2e4t11jxi81aaxnjm6gkbk3fa67n6ux4s9fxnb2k0ekckg"
API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
MODEL = "mimo-v2.5-pro"


# ──────────────────────────────────────────────
# 基础用法: 创建 agent → run
# ──────────────────────────────────────────────

async def basic():
    """最简单的用法 — 指定 workspace 和 skills_dir."""
    agent = ToolsClaw.from_config(
        workspace="./my-project",
        skills_dir="./my-skills",
    )
    result = await agent.run("Hello, 介绍一下你自己")
    print(result.content)


# ──────────────────────────────────────────────
# 从配置文件创建
# ──────────────────────────────────────────────

async def from_config():
    """从 config.json 加载, workspace/skills_dir 可覆盖."""
    agent = ToolsClaw.from_config(
        config_path="./demo-config.json",       # ~/.toolsclaw/config.json
        workspace="./my-project",               # 覆盖 config 中的 workspace
        skills_dir="./my-skills",               # 额外的 skills 目录
    )
    result = await agent.run("列出当前工作区的文件")
    print(result.content)


# ──────────────────────────────────────────────
# 带 Hook 的用法
# ──────────────────────────────────────────────

class LogHook(AgentHook):
    """监听 agent 每一步."""

    async def before_iteration(self, ctx: AgentHookContext) -> None:
        print(f"  [iter {ctx.iteration}] 开始, {len(ctx.messages)} 条消息")

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            print(f"  → 调用 {tc.name}({tc.arguments})")

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        if ctx.tool_calls:
            print(f"  ← 工具返回 {len(ctx.tool_results)} 个结果")
        else:
            print(f"  ✓ 最终回复 ({len(ctx.final_content or '')} chars)")


async def with_hook():
    """带 Hook 监听的用法."""
    agent = ToolsClaw.from_config(workspace="./my-project")
    result = await agent.run(
        "创建一个 hello.py 并运行它",
        hooks=[LogHook()],
    )
    print(result.content)
    print(f"调用的工具: {result.tools_used}")


# ──────────────────────────────────────────────
# 多 agent 实例
# ──────────────────────────────────────────────

async def multi_agent():
    """多个独立 agent, 各有不同 workspace."""
    agent_a = ToolsClaw.from_config(workspace="./project-a")
    agent_b = ToolsClaw.from_config(workspace="./project-b")

    r_a, r_b = await asyncio.gather(
        agent_a.run("创建 a.txt, 内容 'I am A'"),
        agent_b.run("创建 b.txt, 内容 'I am B'"),
    )
    print("A:", r_a.content[:80])
    print("B:", r_b.content[:80])


# ──────────────────────────────────────────────
# 从 AgentRunner 构建 (完全控制)
# ──────────────────────────────────────────────

async def from_runner():
    """手动构造 AgentRunner, 注入自定义工具."""
    from toolsclaw.config import Config, ProviderConfig, ExecConfig
    from toolsclaw.runner import AgentRunner
    from toolsclaw.tool import Tool

    class CalcTool(Tool):
        @property
        def name(self) -> str:
            return "calc"

        @property
        def description(self) -> str:
            return "计算数学表达式"

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            }

        async def execute(self, **kwargs: Any) -> str:
            try:
                return str(eval(kwargs["expression"]))
            except Exception as e:
                return f"Error: {e}"

    config = Config(
        model=MODEL,
        workspace=str(Path.cwd() / "my-project"),
        providers={"mimo": ProviderConfig(api_key=API_KEY, api_base=API_BASE)},
        exec=ExecConfig(enable=True, timeout=30),
    )

    runner = AgentRunner(config, skills_dir="./my-skills")
    runner._registry.register(CalcTool())

    # 从 runner 构建 ToolsClaw
    agent = ToolsClaw.from_runner(runner)
    result = await agent.run("计算 (123 + 456) * 789")
    print(result.content)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "basic":    basic,
        "config":   from_config,
        "hook":     with_hook,
        "multi":    multi_agent,
        "runner":   from_runner,
    }

    name = sys.argv[1] if len(sys.argv) > 1 else "basic"
    if name not in demos:
        print(f"用法: python demo.py [{' | '.join(demos)}]")
        sys.exit(1)

    print(f"▶ {name}\n")
    asyncio.run(demos[name]())
