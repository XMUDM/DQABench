from __future__ import annotations

## When running alone, you need to add
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict

from server.agent.tools.database.Postgre import PG

from pydantic import BaseModel, Field

def state_info(query):
    try:
        index_info = ""
        view_info = ""
        knob_info = ""
        try:
            database = PG()
        except Exception as e:
            return "Database connection failed, please check if there is any connection problem"
        if "index" in query or "索引" in query:
            index_info = str(database.execute_sql("SELECT indexname FROM pg_indexes WHERE schemaname = 'public';"))
        if "view" in query or "视图" in query:
            view_info = str(database.execute_sql("SELECT table_name FROM information_schema.views WHERE table_schema = 'public';"))
        if "knob" in query or "旋钮" in query:
            knob_info = str(database.execute_sql("SHOW ALL;"))
            
        database.close()
        return str(index_info) + '\n' + str(view_info) + "\n" + str(knob_info)
    except Exception as e:
        return str(e) + "Failed to obtain database information. Please make sure that the action input contains index, view or knob."
    
def database_state_info(query: str):
    # model = model_container.MODEL
    # llmchain = LLMDBinfoChain.from_llm(model, verbose=True, prompt=PROMPT)
    # ans = llmchain.run(query)
    res = state_info(query)
    ans = {"text": res}
    return ans

class DBStateInput(BaseModel):
    query: str = Field(description="the database state information you need to know, must one or more in [indexes, views, knobs].")


if __name__ == "__main__":
    database_state_info("index,view,knob")