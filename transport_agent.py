import os
import pdb
import csv
import json
import time
import numpy as np
import pandas as pd

import geocoder
from fuzzywuzzy import fuzz
from util import NC, YELLOW
from pinecone import Pinecone
from pinecone import ServerlessSpec
from base_agent import BaseAgent
from geopy.geocoders import Nominatim

# ------------------------------------------------------------
# tools
# ------------------------------------------------------------
def get_current_location():
    """
    Getting the current location via GPS.
    
    Returns:
        A string representing the current location address.
    """
    # Fallback for the demo
    current_location = "80 Pasir Panjang Road"
    try:
        # Initialize the Nominatim geocoder
        geolocator = Nominatim(user_agent="my_geocoder_app")

        # Get current latitude and longitude
        g = geocoder.ip('me')
        if g.ok:
            print("Your coordinates (lat, lng):", g.latlng)
        else:
            print("Could not retrieve your coordinates.")
        current_lat, current_lon = g.latlng

        # Perform reverse geocoding
        location = geolocator.reverse((current_lat, current_lon))
        if location:
            print("Nearest address:", location.address)
        else:
            print("No address found for these coordinates.")

        return location.address
    except Exception as e:
        print(f"Error getting current location: {e}")
        pdb.set_trace()
    return current_location

def call_function(name, args):
    if name == "get_current_location":
        return get_current_location(**args)
    else:
        pdb.set_trace()

TOOLS = [
    {
        "type": "function",
        "name": "get_current_location",
        "description": "Get the current GPS location of the user.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    },
    {
        "type": "web_search_preview"
    }
]

class TransportAgent(BaseAgent):
    """Agent responsible for handling GrabTransport related queries."""

    def __init__(self, client, model_name):
        super().__init__(client, model_name)

    # ------------------------------------------------------------
    # Pinecone RAG
    # ------------------------------------------------------------
    def load_saved_places(self, csv_path = "data/saved_places.csv"):
        """Load saved locations from CSV file."""
        self.location_database = {}

        try:
            # Load using pandas for better handling
            df = pd.read_csv(csv_path)

            # Convert to dictionary with lowercase keys for case-insensitive matching
            for _, row in df.iterrows():
                name_lower = row['name'].lower()
                if name_lower in self.location_database:
                    print(f"Warning: Duplicate location name '{name_lower}' found in CSV. Using first occurrence.")
                    pdb.set_trace()
                    continue
                self.location_database[name_lower] = row['address']
                
            print(f"Loaded {len(self.location_database)} saved locations from {csv_path}")
            
        except Exception as e:
            print(f"Error loading saved places: {e}")
            pdb.set_trace()
            # Fallback to hardcoded values if CSV loading fails
            self.location_database = {
                "home": "123 Orchard Road, Singapore",
                "work": "One Raffles Place, Singapore",
                "gym": "Fitness First, Bugis Junction, Singapore",
                "school": "NUS, 21 Lower Kent Ridge Rd, Singapore",
                "mall": "VivoCity, 1 HarbourFront Walk, Singapore",
                "airport": "Changi Airport, Singapore",
            }

    def get_embeddings(self, texts_lst):
        """Get embeddings for a list of text using OpenAI's embedding API."""
        if not texts_lst:
            return []
            
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts_lst
        )
        
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def setup_pinecone_index(self):
        """Setup Pinecone index."""
        self.pinecone_index = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index_name = "saved-places"
        self.AWS_REGION = "us-east-1"
        self.spec = ServerlessSpec(cloud="aws", region=self.AWS_REGION)

        # Create index (if it doesn't exist)
        if not self.pinecone_index_name in self.pinecone_index.list_indexes().names():
            print("not self.pinecone_index_name in self.pinecone_index.list_indexes().names()")
            self.pinecone_index.create_index(
                self.pinecone_index_name,
                dimension=1536,
                metric='dotproduct',
                spec=self.spec
            )       

        # Load index
        self.RAG_index = self.pinecone_index.Index(self.pinecone_index_name)
        time.sleep(1)
        index_dict = self.RAG_index.describe_index_stats()
        print(f"Index stats: \n---\n{index_dict}\n---\n")

        # Upsert data (if index is empty)
        if index_dict['total_vector_count'] == 0:
            print("index_dict['total_vector_count'] == 0")
            print("Upserting data to RAG index")
            saved_places_lst = []
            embedding_lst = []
            metadata_lst = []
            for location_name, address in self.location_database.items():
                print(f"location_name = {location_name}")
                embedding_list = self.get_embeddings([location_name])
                print(f"\taddress = {address}")
                print(f"\tlen(embedding_list) = {len(embedding_list)}")

                # Update
                saved_places_lst.append(location_name)
                embedding_lst.append(embedding_list)
                metadata_lst.append({
                    "location_name": location_name,
                    "address": address,
                })
            vectors = list(zip(saved_places_lst, embedding_lst, metadata_lst))
            self.RAG_index.upsert(vectors=vectors)

        # warmup
        # match_metadata = self.query_RAG_index(["parents home"])
    
    def query_RAG_index(self, query_lst, top_k=1, include_metadata=True):
        """
        Returns
            {
                'matches': [
                    {
                        'id': 'parents home',
                        'metadata': {
                            'address': '[Parents Home] 1 Holland Rise, some GCB, Singapore 278123',
                            'location_name': 'parents home'
                        },
                        'score': 0.530264378,
                        'values': []
                    }
                ],
                'namespace': '',
            }
        """
        query_embedding_lst = self.get_embeddings(query_lst)
        # Query the index and return top_k matches.
        result_dict = self.RAG_index.query(
            vector=query_embedding_lst,
            top_k=top_k,
            include_metadata=include_metadata,
        )

        return result_dict

    # ------------------------------------------------------------
    # Generate parameters for the transport agent.
    # ------------------------------------------------------------
    def generate_params(self):
        """Generate parameters for the transport agent."""
        # Setup RAG for Saved Places
        self.load_saved_places()
        self.setup_pinecone_index()

        # Define functions/tools
        self.tools = TOOLS
        
        # Grab Transport RAG
        self.TRANSPORT_SYSTEM_PROMPT = (
            "Extract key information about GrabTransport bookings from user inputs. "
            "User input is an ASR (automatic speech recognition) generated text, so there might be typos, fix them in Singapore's context, where relevant. "
            "For pickup_point and destination_point, extract the location mention exactly as stated, even if it's a saved place like 'home' or 'work'. "
            "However, if pickup_point is not specified in the user query, use the get_current_location function to get the user's current GPS location. "
            "Always include a destination_point in your response."
            "\n\n"
            "If the user asks to go to the 'nearest' or 'closest' location of a certain type (e.g., 'nearest mcdonalds', 'closest starbucks'):"
            "1. First use get_current_location to determine the user's current location."
            "2. Then use find_nearest_location to find the address of the nearest location of that type."
            "3. Use the returned address as the destination_point in your response."
        )
        print(f"self.TRANSPORT_SYSTEM_PROMPT = \n{YELLOW}{self.TRANSPORT_SYSTEM_PROMPT}{NC}")

        # JSON Schema
        json_schema_name = "transport_details"
        transport_json_details = {
            "taxi_type": {
                "type": "string",
                'description': (
                    "JustGrab: The standard and default that has both availablity of both GrabCar and Taxi. "
                        "Standard Taxi: Not a private hire, a metered taxi. "
                        "GrabCar: Private hire, not taxi. "
                        "GrabShare: Cheaper carpooling version, might share with other people. "
                        "GrabHitch: Advanced and cheaper version, not a private hire nor taxi, is for when drivers are on the way. Requires specified timing. "
                    "Defaults to JustGrab."
                ),
                "enum": [
                    "JustGrab",
                    "Standard Taxi",
                    "GrabCar",
                    "GrabShare",
                    "GrabHitch",
                ]
            },
            "pickup_point": {
                'type': "string",
                'description': (
                    "User's starting location. "
                    "If unspecified, system will automatically use GPS to detect current location."
                )
            },
            "destination_point": {
                'type': "string",
                'description': (
                    "User's desired destination. "
                    "It could be a specific address, or a saved places name. "
                )
            },
            "timing": {
                'type': "string",
                'description': (
                    "Only for advanced booking such as GrabHitch, that requires time information."
                    "Exclude the date, only include time."
                    "Defaults to Now, if not specified."
                )
            },
        }
        self.transport_json_schema_format = {
            "format": {
                "type": "json_schema",
                "name": json_schema_name,
                "schema": {
                    "type": "object",
                    "properties": transport_json_details,
                    "required": list(transport_json_details),
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    # ------------------------------------------------------------
    # Resolve saved locations in the database and replace them with actual addresses.
    # ------------------------------------------------------------
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    # Helper function to find the best match for a location
    def find_location_match(self, location_text, verbose=False):
        if not location_text:
            return None
            
        # Try exact match first (case-insensitive)
        location_text = location_text.lower()
        if location_text in self.location_database:
            if verbose:
                print(f"\tExact match: '{location_text}' in self.location_database")
            return self.location_database[location_text]

        # RAG approach with embeddings
        try:
            similarity_threshold = 0.67 # Minimum similarity threshold | Embedding match: 'house' vs 'home': 0.6729 
            RAG_retrieved_metadata = self.query_RAG_index(query_lst=[location_text])
            if RAG_retrieved_metadata:
                """
                    {
                        'matches': [
                            {
                                'id': 'parents home',
                                'metadata': {
                                    'address': '[Parents Home] 1 Holland Rise, some GCB, Singapore 278123',
                                    'location_name': 'parents home'
                                },
                                'score': 0.530264378,
                                'values': []
                            }
                        ],
                        'namespace': '',
                    }
                """
                for match in RAG_retrieved_metadata['matches']:
                    RAG_score = match['score']
                    RAG_retrieved_metadata = match['metadata']
                    print(f"\t\tEmbedding match: '{location_text}' vs '{RAG_retrieved_metadata['location_name']}': {RAG_score:.4f}")
                    if RAG_score >= similarity_threshold:
                        return RAG_retrieved_metadata['address']
        except Exception as e:
            print(f"Error in embedding comparison: {e}")
            pdb.set_trace()
            
        # Try levenshtein ratio measures
        best_match = None
        max_fuzzy_ratio = 0.0
        ratio_threshold = 85
        for saved_name, address in self.location_database.items():
            # levenshtein distance
            current_fuzzy_ratio = fuzz.ratio(location_text, saved_name)
            if (current_fuzzy_ratio > max_fuzzy_ratio) and (current_fuzzy_ratio >= ratio_threshold):
                max_fuzzy_ratio = current_fuzzy_ratio
                best_match = address
                if verbose:
                    print(f"\tRatio match: '{location_text}' ~ '{saved_name}': '{current_fuzzy_ratio}'")
        if best_match:
            return best_match

        # Try partial matching
        for saved_name, address in self.location_database.items():
            # Check if the saved name contains the location text or vice versa
            if saved_name in location_text or location_text in saved_name:
                if verbose:
                    print(f"\tPartial match: '{location_text}' ~ '{saved_name}'")
                return address
            
        return None

    def resolve_saved_locations(self, transport_dict, verbose=False):
        """
        Look up saved locations in the database and replace them with actual addresses.
        
        Args:
            transport_dict: The transport dictionary to process.
            verbose: Whether to print verbose output.
            
        Returns:
            The processed transport dictionary.
        """
        # Check pickup_point
        if transport_dict.get("pickup_point"):
            pickup_address = self.find_location_match(transport_dict["pickup_point"], verbose=verbose)
            if pickup_address:
                if verbose:
                    print(f"\tResolving pickup location: '{transport_dict['pickup_point']}' -> '{pickup_address}'")
                transport_dict["pickup_point"] = pickup_address
            
        # Check destination_point
        if transport_dict.get("destination_point"):
            destination_address = self.find_location_match(transport_dict["destination_point"], verbose=verbose)
            if destination_address:
                if verbose:
                    print(f"\tResolving destination location: '{transport_dict['destination_point']}' -> '{destination_address}'")
                transport_dict["destination_point"] = destination_address
        
        return transport_dict

    # ------------------------------------------------------------
    # Process a transport-related user query.
    # ------------------------------------------------------------
    def process_query_tool_use(self, input_messages, depth=0, verbose=False):
        try:
            print(f"\nprocess_query_tool_use()")
            # Need few shot for better tool use ?
            response_fx = self.client.responses.create(
                model=self.model_name,
                input=input_messages,
                tools=self.tools,
                store=False,
            )

            # Handle function calling
            tool_used = False
            for idx, tool_call in enumerate(response_fx.output):
                # web_search_preview
                if tool_call.type == "web_search_call":
                    print(f"tool_call = {tool_call}")
                    continue
                # is a message
                elif tool_call.type != "function_call":
                    print(f"Tool call type is not function call")
                    input_messages.append({"role": "assistant", "content": tool_call.content[0].text})
                    input_messages.append({"role": "user", "content": "Proceed"})
                    continue
                name = tool_call.name
                args = json.loads(tool_call.arguments)
                result = call_function(name, args)
                # Append tool call and result
                input_messages.append(tool_call)
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result)
                })
                if verbose:
                    print(f"\tdepth={depth} | idx={idx}/{len(response_fx.output)} | [Function call] {name}(**{args}) = {result}")
                tool_used = True
            
            if tool_used:
                print(f"Proceed to the next tool use")
                self.process_query_tool_use(input_messages, depth=depth+1, verbose=verbose)
            
            print(f"End of tool use (depth={depth})")
        except Exception as e:
            error_string = str(e)
            print(f"Error in process_query_tool_use: {error_string}")
            pdb.set_trace()

    def process_query(self, user_query, verbose=False):
        """
        Get transport booking details from a user query.
        
        Args:
            user_query: The user query to process.
            verbose: Whether to print verbose output.
            
        Returns:
            A dictionary with transport booking details.
        """
        try:
            print(f"TransportAgent.process_query({user_query})")

            # Prepare input messages
            input_messages = [
                {"role": "system", "content": self.TRANSPORT_SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]

            self.process_query_tool_use(input_messages, verbose=verbose)
            print(f"process_query(): End of tool use")

            # Perform next inference for JSON schema
            for i in input_messages: print(f"\n{i}")
            response = self.client.responses.create(
                model=self.model_name,
                input=input_messages,
                text=self.transport_json_schema_format,
                store=False,
            )
            if verbose:
                print(f"Response text: {response.output_text}")

            transport_dict = json.loads(response.output_text)
            if verbose:
                print(f"Response dict: {json.dumps(transport_dict, indent=4)}")

            # Process the response to handle saved locations
            transport_dict = self.resolve_saved_locations(transport_dict, verbose)
            if verbose:
                print(f"Transport dict after resolving locations: {json.dumps(transport_dict, indent=4)}")
        except Exception as e:
            error_string = str(e)
            print(f"Error in process_query_tool_use: {error_string}")
            pdb.set_trace()
            
        return transport_dict
