import asyncio
import sys
import time
from toolsclaw import ToolsClaw

async def main():
    t_start = time.perf_counter()

    agent = ToolsClaw.from_config(
        workspace="./my-project",
        skills_dir=r"E:\LLM\toolsclaw\my-project\skill",
        model="mimo-v2.5-pro",
        api_key="tp-cs2e4t11xnb2k0ekckg",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
    )

    print("=" * 60)
    print("Starting PDF to PPT conversion...")
    print("=" * 60)

    result = await agent.run("使用相关技能将工作目录下的2602.12670v3.pdf转换为ppt")

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    content = result.content or "(empty)"
    try:
        print(content)
    except UnicodeEncodeError:
        print(content.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))

    print("\n" + "=" * 60)
    print("STATS")
    print("=" * 60)
    print(f"LLM 轮次:       {result.iterations}")
    print(f"Prompt tokens:  {result.prompt_tokens:,}")
    print(f"Completion:     {result.completion_tokens:,}")
    print(f"Total tokens:   {result.total_tokens:,}")
    print(f"工具调用:       {result.tools_used}")
    print(f"消息数量:       {len(result.messages)}")
    print(f"总耗时:         {elapsed:.2f}s")

asyncio.run(main())
