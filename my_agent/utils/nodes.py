from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from my_agent.utils.tools import tools


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    elif model_name in "gemini":
        model = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-1.5-flash")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


# Define the function that determines whether to continue or not
def should_continue(state):
    # Get the list of response from the state
    response = state.get("response", None)

    # Ensure there are messages to check
    if not response:
        return "end"

    # Check if the last message has a "tool_calls" field
    # Assume tool_calls is a list or flag in the message indicating tool usage
    if not len(response.tool_calls):  # If no tool call, end
        return "end"

    # Otherwise, continue
    return "continue"


system_prompt = "You are an assistant that provides helpful information."


# Define the function that calls the model
def call_model(state, config):
    messages = state.get("messages", [])
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get("configurable", {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # Update the state with the model response
    state.update(response=response)
    return state


# Define the function to execute tools
tool_node = ToolNode(tools)
