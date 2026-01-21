from fastapi import FastAPI
from utils.helpers import intelligent_truncation, agent_repl, code_extractor
from appworld import AppWorld, load_task_ids
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

app = FastAPI()

# NOTE: none of this tested, just a framework

# TODO: validate exact AppWorld input structure with vLLM setup, and exact vLLM output
# test locally with 500m model or llama cpp or something
class ChatMessage(BaseModel):
    """
    Pydantic model replicating minimal setup of openai endpoints
    """
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None

class AppWorldMeta(BaseModel):
    task_id: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    stream: Optional[bool] = False
    appworld: Optional[AppWorldMeta] = None


@app.post("/v1/chat/completions")
async def completions(req: ChatCompletionRequest):
    # need to check if this is applicable
    max_tokens = req.max_tokens or req.max_completion_tokens

    # pull custom metadata
    meta = req.appworld
    task_id = meta.task_id if meta else None

    # first get the raw response from vLLM server
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1",
    )

    # optional intelligent summarization
    if os.getenv("TRUNCATE") == "T":
        messages = intelligent_truncation([m.dict(exclude_none=True) for m in req.messages])
    else:
        messages = [m.dict(exclude_none=True) for m in req.messages]

    response = client.chat.completions.create(
        model=req.model,
        messages=messages,
        temperature=req.temperature,
        max_tokens=max_tokens,
        tools=req.tools,
        tool_choice=req.tool_choice,
        stop=req.stop,
    )

    # now extract the code that would be run by appworld
    code_snippet = code_extractor(response.choices[0].message)

    # pass the code snippet into repl. repl either returns, the input (no problems found)
    # or returns a suggestion
    final_ouput = agent_repl(code_snippet, task_id)

    return response



@app.get("/v1/models")
async def completions():
    return {"message": "null"}