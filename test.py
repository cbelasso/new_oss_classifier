import asyncio

from pydantic import BaseModel

from vllm_server_2 import run_inference_from_list


class Response(BaseModel):
    response: str


async def test():
    prompts = [
        "What is the capital of Canada?",
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
        "What is the capital of Australia?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Mexico?",
        "What is the capital of India?",
    ]

    out = await run_inference_from_list(
        prompts=prompts,
        schema_class=Response,
        model_name="openai/gpt-oss-120b",
        server_url="http://localhost:9001/v1",
    )
    print(out)


asyncio.run(test())
