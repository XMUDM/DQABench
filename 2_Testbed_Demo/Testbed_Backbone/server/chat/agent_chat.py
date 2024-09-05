from langchain.memory import ConversationBufferWindowMemory

from server.agent.custom_agent.ChatGLM3Agent import initialize_glm3_agent
from server.agent.tools_select import tools, tool_names
from server.agent.callbacks import CustomAsyncIteratorCallbackHandler, Status
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from server.agent.custom_template import CustomOutputParser, CustomPromptTemplate
from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE, HISTORY_LEN, Agent_MODEL
from server.utils import wrap_done, get_ChatOpenAI, get_prompt_template
from langchain.chains import LLMChain
from typing import AsyncIterable, Optional
import asyncio
from typing import List
from server.chat.utils import History
import json
from server.agent import model_container
from server.knowledge_base.kb_service.base import get_kb_details


async def agent_chat(query: str = Body(..., description="User Input", examples=["恼羞成怒"]),
                     history: List[History] = Body([],
                                                   description="Historical Dialogue",
                                                   examples=[[
                                                       {"role": "user", "content": "请使用知识库工具查询今天北京天气"},
                                                       {"role": "assistant",
                                                        "content": "使用天气查询工具查询到今天北京多云，10-14摄氏度，东北风2级，易感冒"}]]
                                                   ),
                     stream: bool = Body(False, description="Streaming Output"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM name."),
                     temperature: float = Body(TEMPERATURE, description="LLM sampling temperature", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(None, description="Limit the number of tokens generated by LLM. The default value is None, which represents the maximum value of the model."),
                     prompt_name: str = Body("default",
                                             description="The name of the prompt template to use (configured in configs-prompt_config.py)"),
                     # top_p: float = Body(TOP_P, description="LLM core sampling. Do not set this at the same time as temperature.", gt=0.0, lt=1.0),
                     ):
    history = [History.from_data(h) for h in history]

    async def agent_chat_iterator(
            query: str,
            history: Optional[List[History]],
            model_name: str = LLM_MODELS[0],
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = CustomAsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        ## Pass in global variables to implement agent calls
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
        model_container.DATABASE = {name: details['kb_info'] for name, details in kb_list.items()}

        if Agent_MODEL:
            ## If there is a specified Agent model to complete the task
            model_agent = get_ChatOpenAI(
                model_name=Agent_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=[callback],
            )
            model_container.MODEL = model_agent
        else:
            model_container.MODEL = model

        prompt_template = get_prompt_template("agent_chat", prompt_name)
        prompt_template_agent = CustomPromptTemplate(
            template=prompt_template,
            tools=tools,
            input_variables=["input", "intermediate_steps", "history"]
        )
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=model, prompt=prompt_template_agent)
        # Convert history to agent memory
        memory = ConversationBufferWindowMemory(k=HISTORY_LEN * 2)
        for message in history:
            # Check the role of the message
            if message.role == 'user':
                # Add User Message
                memory.chat_memory.add_user_message(message.content)
            else:
                # Add AI Message
                memory.chat_memory.add_ai_message(message.content)

        if "chatglm3" in model_container.MODEL.model_name:
            agent_executor = initialize_glm3_agent(
                llm=model,
                tools=tools,
                callback_manager=None,
                # Langchain Prompt is not constructed directly here, it is constructed inside the GLM3 agent.
                prompt=prompt_template,
                input_variables=["input", "intermediate_steps", "history"],
                memory=memory,
                verbose=True,
            )
        else:
            agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:", "Observation"],
                allowed_tools=tool_names,
            )
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                                tools=tools,
                                                                verbose=True,
                                                                memory=memory,
                                                                )
        while True:
            try:
                task = asyncio.create_task(wrap_done(
                    agent_executor.acall(query, callbacks=[callback], include_run_info=True),
                    callback.done))
                break
            except:
                pass

        if stream:
            async for chunk in callback.aiter():
                tools_use = []
                # Use server-sent-events to stream the response
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                elif data["status"] == Status.error:
                    tools_use.append("\n```\n")
                    tools_use.append("Tool Name:" + data["tool_name"])
                    tools_use.append("Tool Status: " + "Invocation Failure")
                    tools_use.append("Error: " + data["error"])
                    tools_use.append("Retry")
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.tool_finish:
                    tools_use.append("\n```\n")
                    tools_use.append("Tool Name:" + data["tool_name"])
                    tools_use.append("Tool Status:" + "Invocation Successful")
                    tools_use.append("Tool Input: " + data["input_str"])
                    tools_use.append("Tool Output: " + data["output_str"])
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.agent_finish:
                    yield json.dumps({"final_answer": data["final_answer"]}, ensure_ascii=False)
                else:
                    yield json.dumps({"answer": data["llm_token"]}, ensure_ascii=False)


        else:
            answer = ""
            final_answer = ""
            async for chunk in callback.aiter():
                # Use server-sent-events to stream the response
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                if data["status"] == Status.error:
                    answer += "\n```\n"
                    answer += "Tool Name: " + data["tool_name"] + "\n"
                    answer += "Tool Status:" + "Invocation Failure" + "\n"
                    answer += "Error: " + data["error"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.tool_finish:
                    answer += "\n```\n"
                    answer += "Tool Name:" + data["tool_name"] + "\n"
                    answer += "Tool Status:" + "Invocation Successful" + "\n"
                    answer += "Tool Input: " + data["input_str"] + "\n"
                    answer += "Tool Output: " + data["output_str"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.agent_finish:
                    final_answer = data["final_answer"]
                else:
                    answer += data["llm_token"]

            yield json.dumps({"answer": answer, "final_answer": final_answer}, ensure_ascii=False)
        await task

    return EventSourceResponse(agent_chat_iterator(query=query,
                                                 history=history,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )
