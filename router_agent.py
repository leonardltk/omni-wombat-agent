import json
import pdb
from base_agent import BaseAgent
from util import NC, YELLOW

class RouterAgent(BaseAgent):
    """Agent responsible for detecting user intent and routing to appropriate services."""
    
    def generate_params(self):
        """Generate parameters for the router agent."""
        # Intent Detection
        self.INTENT_DETECTION_PROMPT = (
            "You are a helpful assistant from the ride-hailing service Grab. "
            "Your task is to identify relevant Grab services from user inputs. "
            'Choose only from the following: "GrabTransport", "GrabFood", "GrabPay".\n'
                '- "GrabTransport" : If the user is making a transport booking from pickup to destination.\n'
                '- "GrabFood" : If the user is ordering food.\n'
                '- "GrabPay" : If the user wants to perform a financial transaction.\n'
                '- "null" : If the user query is neither of the above.'
        )
        print(f"self.INTENT_DETECTION_PROMPT = \n{YELLOW}{self.INTENT_DETECTION_PROMPT}{NC}")

        self.router_json_schema_format = {
            "format": {
                "type": "json_schema",
                "name": "service_detection",
                "schema": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "The service that Grab offers.",
                            "enum": [
                                "GrabTransport",
                                "GrabFood",
                                "GrabPay",
                                "null",
                            ]
                        }
                    },
                    "required": ["service"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    
    def get_intent_detection_payload(self, user_query, verbose=False):
        """
        Detect the intent of a user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with the detected service.
        """
        return self.get_payload(
            user_query=user_query,
            system_prompt=self.INTENT_DETECTION_PROMPT,
            json_schema_format=self.router_json_schema_format,
            verbose=verbose
        )
    
    def process_query(self, user_query, verbose=False):
        """
        Process a user query to detect the intent.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with the detected service.
        """
        return self.get_intent_detection_payload(user_query, verbose) 