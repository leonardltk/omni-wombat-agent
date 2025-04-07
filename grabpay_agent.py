import json
from base_agent import BaseAgent
from util import NC, YELLOW

class GrabPayAgent(BaseAgent):
    """Agent responsible for handling GrabPay related queries."""
    
    def generate_params(self):
        """Generate parameters for the GrabPay agent."""
        # Grab Pay
        self.GRABPAY_SYSTEM_PROMPT = (
            "Extract key information about GrabPay transactions from user inputs. "
            "User input is an ASR (automatic speech recognition) generated text, so there might be typos, fix them in Singapore's context, where relevant."
        )
        print(f"self.GRABPAY_SYSTEM_PROMPT = \n{YELLOW}{self.GRABPAY_SYSTEM_PROMPT}{NC}")

        # JSON Schema
        json_schema_name = "grabpay_details"
        grabpay_json_properties = {
            "service": {
                "type": "string",
                "description": (
                    "The service that GrabPay offers. "
                    "Choose only from the following: homescreen, top up, scan to pay, transfer, receive.\n"
                        "- homescreen: If user wants to open the GrabPay homescreen.\n"
                        "- top up: If user wants to top up their GrabPay wallet.\n"
                        "- scan to pay: If user wants to scan to pay with their GrabPay wallet.\n"
                        "- transfer: If user wants to transfer money to a recipient.\n"
                        "- receive: If user wants to receive money from a sender.\n"
                    "Defaults to homescreen."
                ),
                "enum": ["homescreen", "top up", "scan to pay", "transfer", "receive"]
            },
            "recipient": {
                "type": "string",
                "description": (
                    "The recipient of the transfer, only when transfer is detected."
                    "Defaults to empty string."
                ),
            },
        }
        self.grabpay_json_schema_format = {
            "format": {
                "type": "json_schema",
                "name": json_schema_name,
                "schema": {
                    "type": "object",
                    "properties": grabpay_json_properties,
                    "required": list(grabpay_json_properties),
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    
    def get_grabpay_payload(self, user_query, verbose=False):
        """
        Get GrabPay transaction details from a user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with GrabPay transaction details.
        """
        return self.get_payload(
            user_query=user_query,
            system_prompt=self.GRABPAY_SYSTEM_PROMPT,
            json_schema_format=self.grabpay_json_schema_format,
            verbose=verbose
        )
    
    def process_query(self, user_query, verbose=False):
        """
        Process a GrabPay-related user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with GrabPay transaction details.
        """
        print(f"GrabPayAgent.process_query({user_query})")
        return self.get_grabpay_payload(user_query, verbose) 