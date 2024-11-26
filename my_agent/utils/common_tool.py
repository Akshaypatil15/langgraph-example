from dotenv import load_dotenv
from web3 import Web3
import os

# Load environment variables
load_dotenv()


class Web3Manager:
    """
    A manager class for handling Web3 connections to Ethereum networks.
    """

    def __init__(self):
        """
        Initializes the Web3Manager with a provider URL.
        """
        self.provider_url = os.getenv("INFURA_URL")
        self.web3 = self._connect()

    def _connect(self) -> Web3:
        """
        Establishes a connection to the Ethereum network.

        Returns:
            Web3: An initialized Web3 instance.

        Raises:
            ConnectionError: If the connection to the Ethereum network fails.
        """
        web3_instance = Web3(Web3.HTTPProvider(self.provider_url))
        if not web3_instance.is_connected():
            raise ConnectionError(
                f"Failed to connect to Ethereum network at {self.provider_url}"
            )
        return web3_instance

    def get_web3_instance(self) -> Web3:
        """
        Returns the Web3 instance.

        Returns:
            Web3: The Web3 instance.
        """
        return self.web3


# Usage in common_tool.py
web3_manager = Web3Manager()
web3 = web3_manager.get_web3_instance()
