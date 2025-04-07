import json
import pdb
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from router_agent import RouterAgent
from transport_agent import TransportAgent
from food_agent import FoodAgent
from grabpay_agent import GrabPayAgent
from util import NC, BOLD, RED, GREEN, YELLOW, BLUE

class GrabServiceRouter:
    """
    Main service router that handles routing user queries to appropriate service agents.
    This version uses specialized agent classes for each service.
    """
    
    def __init__(self):
        """Initialize the GrabServiceRouter with all specialized agents."""
        load_dotenv(find_dotenv())  # read local .env file
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set in environment"
        
        # Create OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "gpt-4o-mini"
        
        # Initialize agent instances
        self.router_agent = RouterAgent(client=self.client, model_name=self.model_name)
        self.transport_agent = TransportAgent(client=self.client, model_name=self.model_name)
        self.food_agent = FoodAgent(client=self.client, model_name=self.model_name)
        self.grabpay_agent = GrabPayAgent(client=self.client, model_name=self.model_name)
    
    def warmup(self):
        """Warm up the system with a few example queries."""
        self.run("Hi, how are you?")
        self.run("Go to marina bay sands.")
        self.run("Order mcdonalds to my home.")
        self.run("Top up money.")
    
    def routing_agent(self, service_dict, user_query):
        """
        Route the user query to the appropriate agent based on the detected service.
        
        Args:
            service_dict: Dictionary containing the detected service.
            user_query: The original user query.
            
        Returns:
            Dictionary with service and agent response.
        """
        output_dict = {
            'service': service_dict['service'],
            'agent_response': None,
        }

        if service_dict['service'] == "null":
            output_dict['agent_response'] = "I did not understand your request, can you try again?"
        elif service_dict['service'] == "GrabTransport":
            output_dict['agent_response'] = self.transport_agent.process_query(user_query, verbose=True)
        elif service_dict['service'] == "GrabFood":
            output_dict['agent_response'] = self.food_agent.process_query(user_query)
        elif service_dict['service'] == "GrabPay":
            output_dict['agent_response'] = self.grabpay_agent.process_query(user_query)
        else:
            pdb.set_trace()

        return output_dict
    
    def run(self, user_query):
        """
        Main entry point to process a user query.
        
        Args:
            user_query: The user query to process.
            
        Returns:
            Dictionary with service and agent response.
        """
        print(f"run({user_query})")
        
        # Detect intent using the router agent
        service_dict = self.router_agent.process_query(user_query)
        
        # Hand off to the appropriate service agent
        response_dict = self.routing_agent(service_dict, user_query)
        
        return response_dict

# Test
if __name__ == "__main__":
    grab_service_router = GrabServiceRouter()
    
    grab_service_router.run("Hi, how are you?")
    grab_service_router.run("Go to marina bay sands.")
    grab_service_router.run("Order mcdonalds to my home.")
    grab_service_router.run("Top up money.") 