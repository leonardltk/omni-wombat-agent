import json
from base_agent import BaseAgent
from util import NC, YELLOW

class FoodAgent(BaseAgent):
    """Agent responsible for handling GrabFood related queries."""
    
    def generate_params(self):
        """Generate parameters for the food agent."""
        # Grab Food
        self.FOOD_SYSTEM_PROMPT = (
            "Extract key information about GrabFood orders from user inputs. "
            "User input is an ASR (automatic speech recognition) generated text, so there might be typos, fix them in Singapore's context, where relevant."
        )
        print(f"self.FOOD_SYSTEM_PROMPT = \n{YELLOW}{self.FOOD_SYSTEM_PROMPT}{NC}")

        # JSON Schema
        json_schema_name = "food_details"
        food_json_details = {
            "food items": {
                'type': "array",
                'description': (
                    "A list of food or beverage dishes."
                    "Defaults to empty list []"
                ),
                "items": {"type": "string"}
            },
            "sort_by": {
                'type': "string",
                'description': (
                    "Which sorting is specified by user."
                    "Defaults to Recommended."
                ),
                "enum": [
                    "Recommended",
                    "Popularity",
                    "Rating",
                    "Distance",
                ],
            },
            "restrictions": {
                'type': "array",
                'description': (
                    "Any suitable additional filters to apply. "
                        '- "Promo": If user wants current deals or promotions.\n'
                        '- "Halal": If user wants halal options.\n'
                    "Defaults to empty list []"
                ),
                "items": {
                    "type": "string",
                    "enum": [
                        "Promo",
                        "Halal",
                        "",
                    ],
                }
            },
            "delivery_mode": {
                'type': "string",
                'description': (
                    "Whether user wants delivery or to do self pick up instead."
                    "Defaults to Delivery."
                ),
                "enum": [
                    "Delivery",
                    "Self Pickup",
                ],
            },
            "cuisine_type": {
                'type': "array",
                'description': (
                    "A list of desired cuisines."
                    "Defaults to empty list []"
                ),
                "items": {
                    "type": "string",
                    "enum": [
                        "Western", "Pasta", "Rating", "Distance", "Japanese", "Sushi", "Bento", "Noodles", "Seafood",
                        "Salad", "Drinks & Beverages", "Ramen", "Healthy", "Local", "Thai", "Local & Malaysian", "Alcohol",
                        "Beverages", "Chinese", "Korean", "Indian",
                        "",
                    ],
                }
            },
        }
        self.food_json_schema_format = {
            "format": {
                "type": "json_schema",
                "name": json_schema_name,
                "schema": {
                    "type": "object",
                    "properties": food_json_details,
                    "required": list(food_json_details),
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    
    def get_food_payload(self, user_query, verbose=False):
        """
        Get food order details from a user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with food order details.
        """
        return self.get_payload(
            user_query=user_query,
            system_prompt=self.FOOD_SYSTEM_PROMPT,
            json_schema_format=self.food_json_schema_format,
            verbose=verbose
        )
    
    def process_query(self, user_query, verbose=False):
        """
        Process a food-related user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with food order details.
        """
        print(f"FoodAgent.process_query({user_query})")
        return self.get_food_payload(user_query, verbose) 