from agent import graph, AgentState, GraphConfig
from dotenv import load_dotenv
import re
import os

load_dotenv()


def extract_wallet_address(question: str) -> str:
    """
    Extract the Ethereum wallet address from the question.

    Args:
        question (str): The question string containing the wallet address.

    Returns:
        str: The Ethereum wallet address.
    """
    match = re.search(r"0x[a-fA-F0-9]{40}", question)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid Ethereum address found in the question.")


def process_question(question: str, initial_state: AgentState, config: GraphConfig):
    # Extract wallet address from the question
    wallet_address = extract_wallet_address(question)

    # Add the wallet address to the state (e.g., in messages or other fields)
    initial_state = AgentState(
        messages=(
            [{"role": "user", "content": question, "wallet_address": wallet_address}]
        )
    )

    # Invoke the graph with the updated state
    final_state = graph.invoke(initial_state, config=config)

    # Return the final state with the balance
    return final_state["tool_output"]  # Assuming tool_output holds the balance


# Example initial state and config
initial_state = AgentState(messages=[])
config = {"model_name": "openai"}

# Example question
question = f"What is the balance of wallet {os.getenv('WALLET_ADDRESS')}?"

# Process the question
response = process_question(question, initial_state, config)

# Print the response (balance)
print(response)
