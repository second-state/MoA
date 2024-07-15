# Mixture-of-Agents in 50 lines of code
import asyncio
import os
from together import AsyncTogether, Together


client = Together(api_key="NA", base_url="https://gemma-2-9b.us.gaianet.network/v1")
user_prompt = "What are some fun things to do in SF?"
reference_models = [
    ("gemma-2-9b-it-Q5_K_M",AsyncTogether(api_key="NA",base_url="https://gemma-2-9b.us.gaianet.network/v1")),
    ("gemma-2-27b-it-Q5_K_M",AsyncTogether(api_key="NA",base_url="https://gemma-2-27b.us.gaianet.network/v1")),
    ("Meta-Llama-3-8B-Instruct-Q5_K_M",AsyncTogether(api_key="NA",base_url="https://llama-3-8b.us.gaianet.network/v1")),
]
aggregator_model = "gemma-2-9b-it-Q5_K_M"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


async def run_llm(model_and_client):
    """Run a single LLM call with a reference model."""
    model, async_client = model_and_client
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    print(model)
    return response.choices[0].message.content


async def main():
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt},
            {"role": "user", "content": ",".join(str(element) for element in results)},
        ],
        stream=True,
    )

    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


asyncio.run(main())
