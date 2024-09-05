from langchain.tools import Tool
from server.agent.tools import *

## Please note that if you are using AgentLM, you should use the English version here.

# Tool.from_function(
#         func=calculate,
#         name="calculate",
#         description="Useful for when you need to answer questions about simple calculations",
#         args_schema=CalculatorInput,
#     ),
#     Tool.from_function(
#         func=arxiv,
#         name="arxiv",
#         description="A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields.",
#         args_schema=ArxivInput,
#     ),
#     Tool.from_function(
#         func=weathercheck,
#         name="weather_check",
#         description="",
#         args_schema=WhetherSchema,
#     ),
#     Tool.from_function(
#         func=shell,
#         name="shell",
#         description="Use Shell to execute Linux commands",
#         args_schema=ShellInput,
#     ),
#     Tool.from_function(
#         func=search_knowledgebase_complex,
#         name="search_knowledgebase_complex",
#         description="Use Use this tool to search local knowledgebase and get information",
#         args_schema=KnowledgeSearchInput,
#     ),
#     Tool.from_function(
#         func=search_internet,
#         name="search_internet",
#         description="Use this tool to use bing search engine to search the internet",
#         args_schema=SearchInternetInput,
#     ),
#     Tool.from_function(
#         func=wolfram,
#         name="Wolfram",
#         description="Useful for when you need to calculate difficult formulas",
#         args_schema=WolframInput,
#     ),
#     Tool.from_function(
#         func=search_youtube,
#         name="search_youtube",
#         description="use this tools to search youtube videos",
#         args_schema=YoutubeInput,
#     ),
tools = [
    Tool.from_function(
        func = database_structure_info,
        name = "database_structure_info_tool",
        description = "Input the database structure information you need to know, must one or more in [tables, columns, keys]. And then the tool will return the corresponding details.",
        args_schema=DBStructureInput,
    ),
    Tool.from_function(
        func = database_execute_sql,
        name = "execute_sql_tool",
        description = "Input the SQL you want to execute. And the tool will return the corresponding result in the database.",
        args_schema=ExecuteSQLInput,
    ),
    Tool.from_function(
        func = database_workload_info,
        name = "database_workload_info_tool",
        description = "Get the recent workloads including query templates and frequency of the database.",
        args_schema=DBWorkloadInput,
    ),
    Tool.from_function(
        func = database_state_info,
        name = "database_state_info_tool",
        description = "Input the database state information you need to know, must one or more in [indexes, views, knobs]. And then the tool will return the corresponding current states of the database.",
        args_schema=DBStateInput,
    )
]

tool_names = [tool.name for tool in tools]
