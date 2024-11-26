from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_output: str
    response: str
    thread_id: str


# class AgentState:
#     def __init__(self, messages=None, thread_id=None, tool_output="", response=""):
#         # Initialize with an empty list if no messages are provided
#         self.messages = messages or []
#         # Example for tool output
#         self.tool_output = tool_output
#         # Example for model response
#         self.response = response
#         # Add thread_id as an optional field, defaulting to None
#         self.thread_id = thread_id or "abc123"

#     def update(self, tool_output=None, response=None, messages=None):
#         if tool_output is not None:
#             self.tool_output = tool_output
#         if response is not None:
#             self.response = response
#         if messages is not None:
#             self.messages = messages
#         return self  # Return the updated state for chaining

#     def __repr__(self):
#         return f"AgentState(messages={self.messages}, tool_output={self.tool_output}, response={self.response}, thread_id={self.thread_id})"
