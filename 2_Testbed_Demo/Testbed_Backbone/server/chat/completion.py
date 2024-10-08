from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Optional
import asyncio
from langchain.prompts import PromptTemplate

from server.utils import get_prompt_template


async def completion(query: str = Body(..., description="User Input", examples=["恼羞成怒"]),
                     stream: bool = Body(False, description="Streaming Output"),
                     echo: bool = Body(False, description="Echo input in addition to output"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM name."),
                     temperature: float = Body(TEMPERATURE, description="LLM sampling temperature", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="Limit the number of tokens generated by LLM. The default value is None, which represents the maximum value of the model."),
                     # top_p: float = Body(TOP_P, description="LLM core sampling. Do not set this at the same time as temperature.", gt=0.0, lt=1.0),
                     prompt_name: str = Body("default",
                                             description="The name of the prompt template to use (configured in configs-prompt_config.py)"),
                     ):

    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )

        prompt_template = get_prompt_template("completion", prompt_name)
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(prompt=prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return EventSourceResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )
