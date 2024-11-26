from typing import TypedDict, List, Tuple, Union
from functools import lru_cache
import logging

# from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

# from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# added by prash
from utils.tools import *
from langchain.agents import Tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define tools
tools = [
    Tool(
        name="Find_Similar_Plan",
        func=find_similar_plan,
        description="Finds the most similar plan from common plans based on the user's query.",
    ),
    Tool(
        name="Get_ETH_Balance",
        func=get_eth_balance,
        description="Fetch the ETH balance of a wallet.",
    ),
    Tool(
        name="Get_Token_Balance",
        func=get_token_balance,
        description="Fetch the balance of a specific token.",
    ),
    Tool(
        name="Lookup_Token_Address",
        func=lookup_token_address,
        description="Look up the contract address for a given token name.",
    ),
    Tool(
        name="Get_Transaction_Status",
        func=get_transaction_status,
        description="Fetch the status of a blockchain transaction.",
    ),
    Tool(
        name="Get_Recent_Transactions",
        func=get_recent_transactions,
        description="Fetch recent transactions from the latest block.",
    ),
    Tool(
        name="Estimate_Gas_Fee",
        func=estimate_gas_fee,
        description="Estimate current gas fees on the network.",
    ),
    Tool(
        name="Get_Best_Swap_Rate",
        func=get_best_swap_rate,
        description="Fetch the best swap rate from DEX aggregators.",
    ),
    Tool(
        name="Get_NFT_Collection",
        func=get_nft_collection,
        description="Fetch a wallet's NFT collection.",
    ),
    Tool(
        name="Get_NFT_Value",
        func=get_nft_value,
        description="Get the estimated value of an NFT.",
    ),
    Tool(
        name="Get_Historical_Prices",
        func=get_historical_prices,
        description="Retrieve historical price data for a token.",
    ),
    Tool(
        name="Get_Staking_Opportunities",
        func=get_staking_opportunities,
        description="Fetch available staking opportunities for a token.",
    ),
    Tool(
        name="Get_Contract_Data",
        func=get_contract_data,
        description="Fetch data from a smart contract.",
    ),
    Tool(
        name="Get_Governance_Tokens",
        func=get_governance_tokens,
        description="Fetch governance tokens held by a wallet.",
    ),
    Tool(
        name="Get_Tokens_From_Etherscan",
        func=get_tokens_from_etherscan,
        description="Fetch the list of tokens in a wallet using Etherscan.",
    ),
    Tool(
        name="Get_NFTs_From_Etherscan",
        func=get_nfts_from_etherscan,
        description="Fetch the list of NFTs in a wallet using Etherscan.",
    ),
    Tool(
        name="Get_Token_Price_CMC",
        func=get_token_price_cmc,
        description="Fetch the current price of a token using CoinMarketCap.",
    ),
    Tool(
        name="Parse_To_JSON_With_LLM",
        func=parse_to_json_with_llm,
        description="Uses an LLM to parse structured text into JSON format, intelligently fixing formatting issues if needed.",
    ),
]


# Define model cache
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "gpt-4o":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "gpt-4o-mini":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "gpt-3.5-turbo":
        model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model


# Define state structure
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: str


# Define models for planning and action
class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to follow, sorted in order.")


class Response(BaseModel):
    response: str


class Act(BaseModel):
    action: Union[Response, Plan]


# Define preplanner agent
preplanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Preplanner agent. Your task is to retrieve a similar plan from existing embeddings, 
modify it to match the user's request, and provide the adjusted plan as output.
Tools available:
Find_Similar_Plan: Finds the most similar plan from common plans based on the user's query.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

preplanner_llm = ChatOpenAI(model="gpt-4o", temperature=0)
preplanner_agent = create_react_agent(
    preplanner_llm,
    tools=[
        Tool(
            name="Find_Similar_Plan",
            func=find_similar_plan,
            description="Finds the most similar plan from common plans based on the user's query.",
        ),
    ],
    state_modifier=preplanner_prompt,
)


async def preplan_step(state: PlanExecute):
    logging.debug(f"Preplanning step. Input: {state.get('input')}")
    input_text = state.get("input", "")
    if not input_text:
        return {"plan": []}  # Return empty plan if input is missing

    # Task: Retrieve and modify a similar plan
    task_prompt = (
        f"Find and modify a similar plan for the following input: {input_text}"
    )
    response = await preplanner_agent.ainvoke({"messages": [("user", task_prompt)]})
    similar_plan = response["messages"][-1].content
    logging.debug(f"Preplanner produced a similar plan: {similar_plan}")

    # Check for missing information
    if "MISSING INFORMATION" in similar_plan:
        logging.debug("Detected missing information in the similar plan.")
        # Ask the user for the missing information
        ask_prompt = f"The retrieved plan requires additional details: {similar_plan}. What information is missing?"
        user_response = ask_human(ask_prompt)
        logging.debug(f"User provided additional information: {user_response}")

        # Update the plan with the user's response
        similar_plan = similar_plan.replace("MISSING INFORMATION", user_response)
        logging.debug(f"Updated plan after user input: {similar_plan}")

    return {"plan": similar_plan}


# Define prompts
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given plan. \
            
modify the plan to specify which tools should be used in each step. The final step should always use the Parse_To_JSON_With_LLM tool. \

The following tools can be used by the execution agent:
Find_Similar_Plan: Finds the most similar plan from common plans based on the user's query.
Get_ETH_Balance: Fetch the ETH balance of a wallet.
Get_Token_Balance: Fetch the balance of a specific token.
Lookup_Token_Address: Look up the contract address for a given token name.
Get_Transaction_Status: Fetch the status of a blockchain transaction.
Get_Recent_Transactions: Fetch recent transactions from the latest block.
Estimate_Gas_Fee: Estimate current gas fees on the network.
Get_Best_Swap_Rate: Fetch the best swap rate from DEX aggregators.
Get_NFT_Collection: Fetch a wallet's NFT collection.
Get_NFT_Value: Get the estimated value of an NFT.
Get_Historical_Prices: Retrieve historical price data for a token.
Get_Staking_Opportunities: Fetch available staking opportunities for a token.
Get_Contract_Data: Fetch data from a smart contract.
Get_Governance_Tokens: Fetch governance tokens held by a wallet.
Get_Tokens_From_Etherscan: Fetch the list of tokens in a wallet using Etherscan.
Get_NFTs_From_Etherscan: Fetch the list of NFTs in a wallet using Etherscan.
Get_Token_Price_CMC: Fetch the current price of a token using CoinMarketCap.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

replanner_prompt = ChatPromptTemplate.from_template(
    """Update the following plan for the objective. Add steps only if necessary. 
Objective: {input}
Original plan: {plan}
Completed steps: {past_steps}"""
)

# Create planner and replanner
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)
replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

# Pull the prompt for the React agent
from langchain import hub

react_prompt = hub.pull("ih/ih-react-agent-executor")

# Updated Execution Agent Prompt
execution_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an execution agent. Your task is to go through and execute each step of the plan using the tools specified. 
Each step in the plan corresponds to a specific tool that should be utilized to achieve the step's objective. 
Ensure you follow the order of the steps provided and return the output of each tool's execution.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define the execution agent
llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, state_modifier=execution_agent_prompt)


# Define steps
async def plan_step(state: PlanExecute):
    logging.debug(f"Planning step. Input: {state.get('input')}")
    input_text = state.get("input", "")
    if not input_text:
        return {"plan": []}  # Return empty plan if input is missing

    plan = await planner.ainvoke({"messages": [("user", input_text)]})
    logging.debug(f"Generated plan: {plan.steps}")
    return {"plan": plan.steps}


async def execute_step(state: PlanExecute):
    logging.debug(f"Executing step. Current state: {state}")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    if not plan:
        return {"past_steps": past_steps}  # Return unchanged if no plan remains

    current_step = plan[0]
    task_prompt = f"""Task: {current_step}."""
    response = await agent_executor.ainvoke({"messages": [("user", task_prompt)]})
    logging.debug(
        f"Executed step: {current_step}. Response: {response['messages'][-1].content}"
    )
    return {
        "past_steps": past_steps + [(current_step, response["messages"][-1].content)],
        "plan": plan[1:],  # Remove the completed step
    }


async def replan_step(state: PlanExecute):
    logging.debug(f"Replanning step. Current state: {state}")
    updated_plan = await replanner.ainvoke(state)
    if isinstance(updated_plan.action, Response):
        logging.debug(f"Replanner produced a response: {updated_plan.action.response}")
        return {"response": updated_plan.action.response}
    else:
        logging.debug(f"Replanner updated the plan: {updated_plan.action.steps}")
        return {"plan": updated_plan.action.steps}


def should_end(state: PlanExecute):
    if state.get("response"):
        logging.debug(f"Workflow ending: Response available: {state['response']}")
        return END
    elif not state.get("plan"):
        logging.debug("Workflow ending: No steps left in the plan.")
        return END
    else:
        return "agent"
