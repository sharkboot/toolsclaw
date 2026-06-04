import asyncio
import sys
from toolsclaw import ToolsClaw

async def main():
    agent = ToolsClaw.from_config(
        workspace="./my-project",
        skills_dir=r"E:\LLM\toolsclaw\my-project\skill",
        model="mimo-v2.5-pro",
        api_key="tp-cs2e4t11jxi81aaxnjm6gkbk3fa67n6ux4s9fxnb2k0ekckg",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
    )

    print("=" * 60)
    print("Starting PDF to PPT conversion...")
    print("=" * 60)

    result = await agent.run("使用相关技能将工作目录下的2602.12670v3.pdf转换为ppt")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    content = result.content or "(empty)"
    try:
        print(content)
    except UnicodeEncodeError:
        print(content.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))
    print(f"\n工具调用: {result.tools_used}")
    print(f"消息数量: {len(result.messages)}")

asyncio.run(main())
