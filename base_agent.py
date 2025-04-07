from openai import OpenAI
import os
import json
from dotenv import load_dotenv, find_dotenv
from util import NC, YELLOW

class BaseAgent:
    """Base agent class that all specialized agents will inherit from."""
    
    def __init__(self, client=None, model_name="gpt-4o-mini"):
        """
        Initialize the base agent.
        
        Args:
            client: OpenAI client instance. If None, a new client will be created.
            model_name: The model name to use for inference.
        """
        if client is None:
            load_dotenv(find_dotenv())  # read local .env file
            assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set in environment"
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = client
            
        self.model_name = model_name
        self.generate_params()
    
    def generate_params(self):
        """Generate parameters for the agent. Should be overridden by subclasses."""
        pass
    
    def process_query(self, user_query):
        """
        Process a user query. Should be overridden by subclasses.
        
        Args:
            user_query: The user query to process.
            
        Returns:
            The processed result.
        """
        raise NotImplementedError("Subclasses must implement process_query")
    
    def get_payload(self, user_query, system_prompt, json_schema_format, verbose=False):
        """
        Get a payload from the OpenAI API.
        
        Args:
            user_query: The user query to process.
            system_prompt: The system prompt to use.
            json_schema_format: The JSON schema format for the response.
            verbose: Whether to print verbose output.
            
        Returns:
            The processed result as a dictionary.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
            text=json_schema_format
        )
        
        if verbose:
            print(f"Response text: {response.output_text}")
            
        response_dict = json.loads(response.output_text)
        
        if verbose:
            print(f"Response dict: {json.dumps(response_dict, indent=4)}")
            
        return response_dict 