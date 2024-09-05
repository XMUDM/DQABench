# The prompt template uses Jinja2 syntax, which is simply a double brace instead of a single brace for an f-string
# This configuration file supports hot loading, so you do not need to restart the service after modifying the prompt template.

# Variables supported by LLM conversations:
#   - input: indicates user input

# Variables supported by knowledge base and search engine conversations:
#   - context: Concatenated knowledge text from search results
#   - question: Questions raised by users

# Agent session supported variables:

#   - tools: List of available tools
#   - tool_names: List of available tool names
#   - history: Conversation history between the user and Agent
#   - input: User input content
#   - agent_scratchpad: Agent's thinking record

PROMPT_TEMPLATES = {
    "db_chat": {
        "English":
            """Given a raw text input to a \
                language model select the model prompt best suited for the input. \
                You will be given the names of the available prompts and a \
                description of what the prompt is best suited for. \
                You may also revise the original input if you think that revising\
                it will ultimately lead to a better response from the language model.

                << FORMATTING >>
                Return a markdown code snippet with a JSON object formatted to look like:
                ```json
                {{{{
                    "destination": string \ name of the prompt to use or "DEFAULT".
                    "next_inputs": string \ copy the original inputs.
                }}}}
                ```

                REMEMBER: "destination" MUST be one of the candidate prompt \
                names specified below. 

                << CANDIDATE PROMPTS >>
                {destinations}

                << INPUT >>
                {{input}}

                << OUTPUT (remember to include the ```json)>>
            """
    },

    "llm_chat": {
        "default":
            '{{ input }}',

        "with_history":
            'The following is a friendly conversation between a human and an AI. '
            'The AI is talkative and provides lots of specific details from its context. '
            'If the AI does not know the answer to a question, it truthfully says it does not know.\n\n'
            'Current conversation:\n'
            '{history}\n'
            'Human: {input}\n'
            'AI:',

        "py":
            'You are a smart code helper, please write simple py code for me. \n'
            '{{ input }}',
    },


    "knowledge_base_chat": {
        "default":
            '<Instruction> Answer questions in a concise and professional manner, based on the information you already know. If you can\'t get an answer from it, say \"This question cannot be answered based on known information.\"'
            'No fabrications are allowed in the answers. Please use Chinese for the answers. </instruction>\n'
            '<Known information>{{context}}</known information>\n'
            '<question>{{ question }}</question>\n',

        "text":
            '<Instruction> Answer questions in a concise and professional manner, based on the information you already know. If you cannot get an answer from it, please say "This question cannot be answered based on the known information", and the answer should be in Chinese. </instruction>\n'
            '<Known information>{{ context }}</Known information>\n'
            '<question>{{ question }}</question>\n',

        "empty":  # Used when the knowledge base cannot be found
            'Please answer my question:\n'
            '{{ question }}\n\n',
    },


    "search_engine_chat": {
        "default":
            '<Instruction> This is the Internet information I have searched. Please extract and adjust the information according to this information and answer the question succinctly.'
            'If you can\'t get an answer from it, say \"No content can be found to answer the question.\" </instruction>\n'
            '<Known information>{{context}}</known information>\n'
            '<question>{{ question }}</question>\n',

        "search":
            '<Instruction> Answer questions in a concise and professional manner, based on the information you already know. If you cannot get an answer from it, please say "This question cannot be answered based on the known information", and the answer should be in Chinese. </instruction>\n'
            '<Known information>{{ context }}</Known information>\n'
            '<question>{{ question }}</question>\n',
    },


    "agent_chat": {
        "default":
            'Answer the following questions as best you can. If it is in order, you can use some tools appropriately. '
            'You have access to the following tools:\n\n'
            '{tools}\n\n'
            'Use the following format:\n'
            'Question: the input question you must answer1\n'
            'Thought: you should always think about what to do and what tools to use.\n'
            'Action: the action to take, should be one of [{tool_names}]\n'
            'Action Input: the input to the action\n'
            'Observation: the result of the action\n'
            '... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n'
            'Thought: I now know the final answer\n'
            'Final Answer: the final answer to the original input question\n'
            'Begin!\n\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}\n',

        "ChatGLM3":
            'You can answer using the tools, or answer directly using your knowledge without using the tools. '
            'Respond to the human as helpfully and accurately as possible.\n'
            'You have access to the following tools:\n'
            '{tools}\n'
            'Use a json blob to specify a tool by providing an action key (tool name) '
            'and an action_input key (tool input).\n'
            'Valid "action" values: "Final Answer" or  [{tool_names}]'
            'Provide only ONE action per $JSON_BLOB, as shown:\n\n'
            '```\n'
            '{{{{\n'
            '  "action": $TOOL_NAME,\n'
            '  "action_input": $INPUT\n'
            '}}}}\n'
            '```\n\n'
            'Follow this format:\n\n'
            'Question: input question to answer\n'
            'Thought: consider previous and subsequent steps\n'
            'Action:\n'
            '```\n'
            '$JSON_BLOB\n'
            '```\n'
            'Observation: action result\n'
            '... (repeat Thought/Action/Observation N times)\n'
            'Thought: I know what to respond\n'
            'Action:\n'
            '```\n'
            '{{{{\n'
            '  "action": "Final Answer",\n'
            '  "action_input": "Final response to human"\n'
            '}}}}\n'
            'Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. '
            'Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}',
    }
}

DB_PROMPT_TEMPLATE = {}


DB_PROMPT_TEMPLATE["general"] = """You are an expert in the field of general database issues, which do not involve specific database instances and are related to Opengauss or gaussDB.
Do not allow any fabrications to be added to the answer. 
    
    Begin!
    History: {history}
    Question: {input}
"""

DB_PROMPT_TEMPLATE["gauss"] = """You are an expert in the field of database issues, which are related to Opengauss and gaussDB databases.
    Answer questions in a concise and professional manner based on the information in "Knowledge", which is from Opengauss and gaussDB documents. 
    Do not allow any fabrications to be added to the answer. 
    
    Begin!
    History: {history}
    Question: {input}
    Knowledge: {knowledge}
"""

DB_PROMPT_TEMPLATE["management"] = """You are an management expert in the specific database instance "OurDB", capable of using tools to extract information from that database.
    Your role is to assist users in addressing issues that may arise during the operation and maintenance of this particular database.
    Your responses should not only draw on expertise in the database field but also provide specific solutions based on details such as tables, columns, constraints, workload information, and other relevant database specifics.

    The tools you can use:
    {tools}

    Use the following format:
    Question: the input question you must answer.
    Thought: you should always think about what to do and what tools to use.
    Action: the action to take, should be one of [{tool_names}] and does not contain any other tokens.
    Action Input: the input to the action.
    Observation: the result of the action.
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times. If you need some information from the user, you can break it and ask on "Final Answer".)
    Final Answer: the final answer to the original input question when you have sufficient knowledge from Observation.

    Begin!
    history: {history}
    Question: {input}
    Thought: {agent_scratchpad}
"""

DB_PROMPT_TEMPLATE["analysis"] = """You are a data analysis expert in the specific database instance "OurDB". You can handle natural language queries from users related to the database and translate them into executable SQL statements.
    You use the "Execute_SQL_Tool" to run these statements and communicate the analysis results to the users. 
    To generate valid SQL statements, you can utilize tools to query information about tables, columns, and constraints in the database:

    {tools}

    Use the following format:
    Question: the input question you must answer.
    Thought: you should always think about what to do and what tools to use.
    Action: the action to take, should be one of [{tool_names}]. aiming to generate a valid SQL statement.
    Action Input: the input to the action.
    Observation: the result of the action.
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times.)
    Final SQL: the final generated SQL to answer user's question.
    SQL Oberservation: the result of executing the SQL statement using "Execute_SQL_Tool".
    Final Answer: the final answer to the original input question from Observation.

    Begin!
    history: {history}
    Question: {input}
    Thought: {agent_scratchpad}
"""

DB_PROMPT_TEMPLATE["normal"] = """
    If the user politely greets you, please politely reply to the user and tell them that you are a database specific intelligent assistant. 

    If the user raises questions unrelated to the database, please politely refuse to answer their questions.

    Begin!
    history: {history}
    Question: {input}
"""
