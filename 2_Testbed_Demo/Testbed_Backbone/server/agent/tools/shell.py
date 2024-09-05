# LangChain's Shell Tool
from pydantic import BaseModel, Field
from langchain.tools import ShellTool
def shell(query: str):
    tool = ShellTool()
    return tool.run(tool_input=query)

class ShellInput(BaseModel):
    query: str = Field(description="A Shell command that can be run on the Linux command line")