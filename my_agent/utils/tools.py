from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from my_agent.utils.common_tool import web3


# Define the input schema using Pydantic
class WalletBalanceInput(BaseModel):
    wallet_address: str


# Define the tool as a structured tool
def get_wallet_balance(inputs: WalletBalanceInput) -> str:
    """
    Fetches the Ethereum (ETH) balance of a specified wallet address.

    Args:
        inputs (WalletBalanceInput): A Pydantic model with the Ethereum wallet address.

    Returns:
        str: A message with the ETH balance or an error message if the operation fails.
    """
    wallet_address = inputs.wallet_address
    try:
        # Fetch balance
        balance_wei = web3.eth.get_balance(wallet_address)
        # Convert Wei to ETH
        balance_eth = web3.from_wei(balance_wei, "ether")
        return f"The ETH balance for {wallet_address} is {balance_eth} ETH"
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


# Wrap the function in a StructuredTool
wallet_balance_tool = StructuredTool.from_function(
    func=get_wallet_balance,
    name="get_wallet_balance",
    description="Fetch the Ethereum wallet balance. Provide a wallet address in hexadecimal format.",
)

# Add tools
tools = [wallet_balance_tool]
