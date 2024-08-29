import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )
app = FastAPI(title="Reflection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    min_p: Optional[float] = 0.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., example="chatcmpl-123")
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completion(request: ChatCompletionRequest):
    if request.model != "reflection_v4_70b":
        raise HTTPException(status_code=400, detail="Invalid model")
    # Add the system prompt to the beginning of the messages
    REFLECT = '''You are a world-class AI system called Llama built by Meta, capable of complex reasoning and reflection. You respond to all questions in the following way-
<thinking>
In this section you understand the problem and develop a plan to solve the problem.

For easy problems-
Make a simple plan and use COT

For moderate to hard problems-
1. Devise a step-by-step plan to solve the problem. (don't actually start solving yet, just make a plan)
2. Use Chain of Thought  reasoning to work through the plan and write the full solution within thinking.

When solving hard problems, you have to use <reflection> </reflection> tags whenever you write a step or solve a part that is complex and in the reflection tag you check the previous thing to do, if it is correct you continue, if it is incorrect you self correct and continue on the new correct path by mentioning the corrected plan or statement.
Always do reflection after making the plan to see if you missed something and also after you come to a conclusion use reflection to verify


</thinking>

<output>
In this section, provide the complete answer for the user based on your thinking process. Do not refer to the thinking tag. Include all relevant information and keep the response somewhat verbose, the user will not see what is in the thinking tag.
</output>'''

    new_messages = []
    for messg in request.messages:
        if messg.role != "system":
            new_messages.append(messg)
        
    messages = [Message(role="system", content=REFLECT)] + request.messages
    
    try:
        # Forward the request to the OpenAI API
        response = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            n=request.n,
            stream=request.stream,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            user=request.user
        )

        # Convert the OpenAI response to our response model
        return ChatCompletionResponse(
            id=response.id,
            object=response.object,
            created=response.created,
            model="reflection_v4_70b",
            choices=[
                Choice(
                    index=choice.index,
                    message=Message(role=choice.message.role, content="<thinking>"+choice.message.content.replace("Sonnet","").replace("Claude","").replace("Anthropic","").split("<thinking>")[1]),
                    finish_reason=choice.finish_reason
                )
                for choice in response.choices
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate completion")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
