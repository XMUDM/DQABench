# Langchain's own Wolfram Alpha API wrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from pydantic import BaseModel, Field
wolfram_alpha_appid = "your key"
def wolfram(query: str):
    wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=wolfram_alpha_appid)
    ans = wolfram.run(query)
    return ans

class WolframInput(BaseModel):
    location: str = Field(description="Specific problems that require calculations")