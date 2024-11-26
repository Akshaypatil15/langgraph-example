import math
import re

import numexpr

# from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


# Crypto Tools

from web3 import Web3
from eth_account import Account
import os
import requests
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Ensure Web3 is initialized
infura_url = os.getenv("INFURA_URL")
web3 = Web3(Web3.HTTPProvider(infura_url))

# ERC20 ABI (You will need to provide this for token interaction)
erc20_abi = "[...]"  # Placeholder for the actual ERC20 ABI


### Wallet & Balance Queries ###
def get_eth_balance(wallet_address):
    try:
        balance = web3.eth.get_balance(wallet_address)  # Directly use the address
        return f"The ETH balance for {wallet_address} is {balance} ETH"
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


def get_token_balance(wallet_address, token_address):
    # Define the ERC20 ABI (minimum needed for balanceOf)
    erc20_abi = [
        {
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function",
        }
    ]
    try:
        # Convert both the wallet and token addresses to checksum addresses
        wallet_address = Web3.to_checksum_address(wallet_address)
        token_address = Web3.to_checksum_address(token_address)

        # Initialize contract with the token address and ERC20 ABI
        contract = web3.eth.contract(address=token_address, abi=erc20_abi)

        # Call the balanceOf function to get the token balance
        balance = contract.functions.balanceOf(wallet_address).call()

        # Assuming token uses 18 decimals by default, adjust it accordingly.
        balance_in_tokens = balance / (10**18)  # Adjust based on actual token decimals
        return f"The balance for {wallet_address} is {balance_in_tokens} tokens"
    except Exception as e:
        return f"Error fetching token balance: {str(e)}"


### Transaction Queries ###
def get_transaction_status(tx_hash):
    try:
        receipt = web3.eth.getTransactionReceipt(tx_hash)
        if receipt is None:
            return "Transaction is pending"
        elif receipt["status"] == 1:
            return "Transaction successful"
        else:
            return "Transaction failed"
    except Exception as e:
        return f"Error fetching transaction status: {str(e)}"


def get_recent_transactions(wallet_address):
    try:
        block = web3.eth.get_block("latest")
        return block["transactions"][
            :10
        ]  # Fetch recent transactions (you can extend this)
    except Exception as e:
        return f"Error fetching recent transactions: {str(e)}"


def estimate_gas_fee():
    try:
        return web3.eth.gas_price
    except Exception as e:
        return f"Error fetching gas fees: {str(e)}"


### Swap & DeFi Queries ###
def perform_token_swap(wallet_address, token_from, token_to, amount):
    try:
        # Placeholder for actual token swap logic using a DEX like Uniswap
        tx_hash = "0x123456789abcdef"  # Dummy transaction hash
        return f"Swap transaction submitted: {tx_hash}"
    except Exception as e:
        return f"Error performing swap: {str(e)}"


def get_best_swap_rate(token_from, token_to):
    try:
        # Call a price aggregator API like 1inch, matcha, etc.
        return {"rate": "Best rate fetched from DEX aggregators"}
    except Exception as e:
        return f"Error fetching swap rate: {str(e)}"


### NFT Queries ###
def get_nft_collection(wallet_address):
    try:
        # Placeholder for fetching NFT collection
        return {"nfts": "List of NFTs owned by the wallet"}
    except Exception as e:
        return f"Error fetching NFT collection: {str(e)}"


def get_nft_value(nft_address):
    try:
        # Placeholder for querying NFT price
        return {"nft_value": "Estimated value of the NFT"}
    except Exception as e:
        return f"Error fetching NFT value: {str(e)}"


def get_historical_prices(token_address):
    try:
        # Placeholder for fetching historical price data
        return {"historical_prices": "Historical prices data"}
    except Exception as e:
        return f"Error fetching historical prices: {str(e)}"


### Staking & Yield Farming ###
def get_staking_opportunities(token_address):
    try:
        # Query DeFi platforms for staking pools
        return {"staking_options": "List of staking opportunities"}
    except Exception as e:
        return f"Error fetching staking opportunities: {str(e)}"


def calculate_staking_yield(token_address, amount):
    try:
        # Fetch staking rates and calculate yield
        return {"apy": "Calculated APY based on staking"}
    except Exception as e:
        return f"Error calculating staking yield: {str(e)}"


### Security & Wallet Management ###
def create_wallet():
    try:
        account = Account.create()
        return {"address": account.address, "private_key": account.privateKey.hex()}
    except Exception as e:
        return f"Error creating wallet: {str(e)}"


def get_private_key(wallet_address):
    # Not recommended for production use, as exposing private keys is dangerous.
    return "Private key management should be done securely."


### Contract Queries ###
def call_smart_contract_function(contract_address, function_name, *args):
    try:
        contract = web3.eth.contract(
            address=contract_address, abi=erc20_abi
        )  # Adjust ABI as needed
        func = getattr(contract.functions, function_name)
        tx_hash = func(*args).transact({"from": web3.eth.accounts[0]})
        return f"Function {function_name} called, transaction: {tx_hash}"
    except Exception as e:
        return f"Error calling smart contract function: {str(e)}"


def get_contract_data(contract_address, function_name):
    try:
        contract = web3.eth.contract(
            address=contract_address, abi=erc20_abi
        )  # Adjust ABI as needed
        data = getattr(contract.functions, function_name)().call()
        return data
    except Exception as e:
        return f"Error fetching contract data: {str(e)}"


### Cross-Chain Queries ###
def bridge_tokens(wallet_address, token_address, amount, target_chain):
    try:
        # Interact with cross-chain bridges
        return {"status": "Bridge transaction submitted"}
    except Exception as e:
        return f"Error bridging tokens: {str(e)}"


### Governance & Voting ###
def vote_on_proposal(proposal_id, wallet_address, vote_choice):
    try:
        # Logic to vote on governance proposals
        return {"status": f"Vote submitted for proposal {proposal_id}"}
    except Exception as e:
        return f"Error submitting vote: {str(e)}"


def get_governance_tokens(wallet_address):
    try:
        # Query the balance of governance tokens in the wallet
        return {"tokens": "List of governance tokens held by wallet"}
    except Exception as e:
        return f"Error fetching governance tokens: {str(e)}"


import requests


def lookup_token_address(token_symbol):
    """
    Lookup the Ethereum contract address for a given token symbol using CoinMarketCap API.

    Args:
        token_symbol (str): The symbol of the token (e.g., "DAI", "USDC").

    Returns:
        str: The contract address of the token or an error message.
    """
    try:
        # Validate input
        if not token_symbol:
            return "Error: Token symbol cannot be empty."

        # Prepare API request
        api_key = os.getenv("COINMARKETCAP_API_KEY")
        if not api_key:
            return "Error: CoinMarketCap API key is not found in environment variables."

        # Step 1: Get CoinMarketCap ID for the given symbol
        url_map = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
        }
        params_map = {"symbol": token_symbol.upper()}

        response_map = requests.get(url_map, headers=headers, params=params_map)

        # Check if map request was successful
        if response_map.status_code == 200:
            data_map = response_map.json().get("data", [])
            if data_map:
                token_id = data_map[0]["id"]
            else:
                return f"No information found for token '{token_symbol}'."
        else:
            return f"Error: Unable to retrieve token ID (status code {response_map.status_code})."

        # Step 2: Use the ID to get detailed info, including contract address
        url_info = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/info"
        params_info = {"id": token_id}

        response_info = requests.get(url_info, headers=headers, params=params_info)

        # Handle the response
        if response_info.status_code == 200:
            data_info = response_info.json().get("data", {})
            if str(token_id) in data_info:
                platforms = data_info[str(token_id)].get("platforms", {})
                eth_address = platforms.get("ethereum")
                if eth_address:
                    return eth_address
                else:
                    return (
                        f"No Ethereum contract address available for '{token_symbol}'."
                    )
            else:
                return f"No detailed information found for token ID '{token_id}'."
        else:
            return f"Error: Unable to retrieve token data (status code {response_info.status_code})."

    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def get_tokens_from_etherscan(wallet_address):
    """
    Get the list of ERC20 tokens held by a given Ethereum wallet address using the Etherscan API.

    Args:
        wallet_address (str): The Ethereum wallet address.

    Returns:
        str: A formatted list of tokens and their balances.
    """
    try:
        # Load Etherscan API key from environment
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return "Error: Etherscan API key is not found in environment variables."

        # Etherscan API endpoint for getting token balances (ERC20)
        url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={wallet_address}&startblock=0&endblock=99999999&sort=asc&apikey={etherscan_api_key}"

        # Make the request to Etherscan API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json().get("result", [])
            if not data:
                return f"No tokens found in wallet {wallet_address}"

            # Prepare the list of tokens and balances
            token_balances = {}
            for token in data:
                token_name = token["tokenName"]
                token_symbol = token["tokenSymbol"]
                token_contract = token["contractAddress"]
                token_value = int(token["value"]) / (10 ** int(token["tokenDecimal"]))

                if token_contract not in token_balances:
                    token_balances[token_contract] = {
                        "name": token_name,
                        "symbol": token_symbol,
                        "balance": 0,
                    }

                token_balances[token_contract]["balance"] += token_value

            # Format the output
            token_list = [
                f"{info['symbol']}: {info['balance']:.4f}"
                for info in token_balances.values()
            ]
            return f"Tokens in {wallet_address}:\n" + "\n".join(token_list)
        else:
            return f"Error: Unable to retrieve token data (status code {response.status_code})."

    except Exception as e:
        return f"Error fetching token balances: {str(e)}"


def get_nfts_from_etherscan(wallet_address):
    """
    Get the list of NFTs held by a given Ethereum wallet address using the Etherscan API.

    Args:
        wallet_address (str): The Ethereum wallet address.

    Returns:
        str: A formatted list of NFTs.
    """
    try:
        # Load Etherscan API key from environment
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return "Error: Etherscan API key is not found in environment variables."

        # Etherscan API endpoint for getting ERC721 token transfers
        url = f"https://api.etherscan.io/api?module=account&action=tokennfttx&address={wallet_address}&startblock=0&endblock=99999999&sort=asc&apikey={etherscan_api_key}"

        # Make the request to Etherscan API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            nfts = response.json().get("result", [])
            if not nfts:
                return f"No NFTs found in wallet {wallet_address}"

            # Prepare the list of NFTs
            nft_list = []
            for nft in nfts:
                contract = nft["contractAddress"]
                token_id = nft["tokenID"]
                name = nft.get("tokenName", "Unknown NFT")
                symbol = nft.get("tokenSymbol", "NFT")
                nft_list.append(
                    f"{name} (Symbol: {symbol}, Token ID: {token_id}, Contract: {contract})"
                )

            return f"NFTs in {wallet_address}:\n" + "\n".join(nft_list)
        else:
            return f"Error: Unable to retrieve NFT data (status code {response.status_code})."

    except Exception as e:
        return f"Error fetching NFTs: {str(e)}"


# Load and preprocess plans from a text file
def load_common_plans(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "common_plans.txt")
    with open(filepath, "r") as file:
        plans = [line.strip() for line in file]
    return plans


# Initialize OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Load the plans and wrap them into Document objects
plans = load_common_plans()
documents = [Document(page_content=plan) for plan in plans]

# Create a FAISS index for similarity search
faiss_index = FAISS.from_documents(documents, embedding_model)


def find_similar_plan(user_input: str):
    """
    Finds the most similar plan from a list of common plans based on the user's query.
    """
    try:
        # Search for the most similar plan using FAISS
        search_results = faiss_index.similarity_search(user_input, k=1)
        best_match_plan = search_results[0].page_content

        return f"Most similar plan: {best_match_plan}"

    except Exception as e:
        return f"Error finding similar plan: {str(e)}"


import time


def perform_uniswap_swap(
    private_key,
    token_in,
    token_out,
    amount_in,
    amount_out_min,
    wallet_address,
    gas_price,
):
    """
    Perform a token swap on Uniswap v3.

    Args:
        private_key (str): The private key of the wallet initiating the swap.
        token_in (str): The address of the token being swapped from.
        token_out (str): The address of the token being swapped to.
        amount_in (int): The amount of token_in to swap (in the smallest unit, e.g., wei).
        amount_out_min (int): The minimum amount of token_out expected (slippage tolerance).
        wallet_address (str): The address of the wallet initiating the swap.
        gas_price (int): The gas price to be used for the transaction.

    Returns:
        str: Transaction hash if the transaction is successful, otherwise an error message.
    """
    try:
        # Ensure Web3 is initialized
        infura_url = os.getenv("INFURA_URL")
        web3 = Web3(Web3.HTTPProvider(infura_url))

        # Check if Web3 is connected
        if not web3.is_connected():
            return "Error: Unable to connect to the Ethereum network."

        # Uniswap Router Contract Address and ABI
        UNISWAP_ROUTER_ADDRESS = (
            "0xE592427A0AEce92De3Edee1F18E0157C05861564"  # Uniswap v3 Router address
        )
        UNISWAP_ROUTER_ABI = [
            {
                "inputs": [
                    {"internalType": "address", "name": "tokenIn", "type": "address"},
                    {"internalType": "address", "name": "tokenOut", "type": "address"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    {"internalType": "address", "name": "recipient", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {
                        "internalType": "uint256",
                        "name": "amountOutMinimum",
                        "type": "uint256",
                    },
                    {
                        "internalType": "uint160",
                        "name": "sqrtPriceLimitX96",
                        "type": "uint160",
                    },
                ],
                "name": "exactInputSingle",
                "outputs": [
                    {"internalType": "uint256", "name": "amountOut", "type": "uint256"}
                ],
                "stateMutability": "payable",
                "type": "function",
            }
        ]

        # Set up the Uniswap router contract
        uniswap_router = web3.eth.contract(
            address=UNISWAP_ROUTER_ADDRESS, abi=UNISWAP_ROUTER_ABI
        )

        # Convert wallet and token addresses to checksum format
        wallet_address = web3.to_checksum_address(wallet_address)
        token_in = web3.to_checksum_address(token_in)
        token_out = web3.to_checksum_address(token_out)

        # Set up transaction parameters
        fee = 3000  # Uniswap v3 fee tier (e.g., 0.3% fee)
        deadline = int(time.time()) + 600  # 10 minutes from now
        sqrt_price_limit_x96 = 0  # No price limit

        # Get the nonce
        nonce = web3.eth.get_transaction_count(wallet_address)

        # Present transaction details to the user
        print("Transaction Details:")
        print(f"Token In: {token_in}")
        print(f"Token Out: {token_out}")
        print(f"Amount In: {amount_in}")
        print(f"Minimum Amount Out: {amount_out_min}")
        print(f"Gas Price: {gas_price} wei")
        print(f"Wallet Address: {wallet_address}")

        # Ask for user confirmation
        confirm = input("Do you want to proceed with the transaction? (yes/no): ")
        if confirm.lower() != "yes":
            return "Transaction cancelled by user."

        # Create the transaction
        transaction = uniswap_router.functions.exactInputSingle(
            token_in,
            token_out,
            fee,
            wallet_address,
            deadline,
            amount_in,
            amount_out_min,
            sqrt_price_limit_x96,
        ).build_transaction(
            {
                "chainId": 1,  # Mainnet
                "gas": 250000,
                "gasPrice": gas_price,  # Use provided gas price
                "nonce": nonce,
                "value": 0,  # Since we're swapping ERC-20 tokens, no ETH value is needed
            }
        )

        # Sign the transaction
        signed_txn = web3.eth.account.sign_transaction(
            transaction, private_key=private_key
        )

        # Send the transaction
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)

        # Wait for the transaction receipt
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        # Check if the transaction was successful
        if tx_receipt["status"] == 1:
            return (
                f"Swap transaction successful, transaction hash: {web3.to_hex(tx_hash)}"
            )
        else:
            return "Error: Transaction failed."

    except AttributeError as e:
        print(f"AttributeError: {e}")
        return f"Error performing swap on Uniswap: {e}"

    except Exception as e:
        print(f"General Exception: {e}")
        return f"Error performing swap on Uniswap: {e}"

    # Define the get_token_price_cmc function


def get_token_price_cmc(token_symbol):
    """
    Fetch the current price of a token using CoinMarketCap API.

    Args:
        token_symbol (str): The symbol of the token (e.g., "BTC", "ETH", "USDC").

    Returns:
        str: The current price of the token or an error message.
    """
    try:
        # Validate input
        if not token_symbol:
            return "Error: Token symbol cannot be empty."

        # Load API key from environment
        api_key = os.getenv("COINMARKETCAP_API_KEY")
        if not api_key:
            return "Error: CoinMarketCap API key is not found in environment variables."

        # Prepare API request
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
        }
        params = {"symbol": token_symbol.upper()}

        # Make the request to CoinMarketCap API
        response = requests.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json().get("data", {})
            if token_symbol.upper() in data:
                price = data[token_symbol.upper()]["quote"]["USD"]["price"]
                return (
                    f"The current price of {token_symbol.upper()} is ${price:.2f} USD."
                )
            else:
                return f"No pricing information found for token '{token_symbol}'."
        else:
            return f"Error: Unable to retrieve token price (status code {response.status_code})."

    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def ask_human(prompt: str) -> str:
    """
    Interact with the user to gather additional information.

    Args:
        prompt (str): The question or prompt to display to the user.

    Returns:
        str: The user's input.
    """
    try:
        # Display the prompt to the user and get their response
        user_input = input(f"{prompt}\n> ")
        return user_input.strip()
    except Exception as e:
        return f"Error during user input: {str(e)}"


from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm_for_parsing = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt template for JSON parsing
json_parsing_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a JSON parsing assistant. Your task is to convert the given text into a valid JSON object. 
If the input cannot be parsed directly, intelligently fix any issues to produce a valid JSON output.
Ensure that the result follows proper JSON formatting.""",
        ),
        ("user", "{input_text}"),
    ]
)


# Function to parse input to JSON using the LLM
def parse_to_json_with_llm(input_text: str) -> str:
    """
    Use an LLM to parse structured output into valid JSON format.

    Args:
        input_text (str): The structured text to be converted to JSON.

    Returns:
        str: The JSON representation or an error message if parsing fails.
    """
    try:
        # Format the input into the LLM prompt
        response = llm_for_parsing.predict(
            input=json_parsing_prompt.format_prompt(input_text=input_text)
        )
        return response
    except Exception as e:
        return f"Error: Failed to parse input into JSON using LLM. Details: {str(e)}"
