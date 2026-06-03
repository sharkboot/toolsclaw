import asyncio
from toolsclaw import ToolsClaw

async def main():
    agent = ToolsClaw.from_config(
        workspace="./my-project",
        skills_dir=r"E:\skill",
        model="mimo-v2.5-pro",
        api_key="tp-cs2e4t11jxi81aaxnjm6gkbk3fa67n6ux4s9fxnb2k0ekckg",
        api_base="https://token-plan-cn.xiaomimimo.com/v1",
    )
    result = await agent.run("Hello, 介绍一下你自己 有哪些工具")
    print(result.content)
    print(f"工具调用: {result.tools_used}")

asyncio.run(main())
