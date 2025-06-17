import gradio as gr
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import random
import argparse

color_green = "\033[92m"
color_reset = "\033[0m"
print(f"{color_green}MCP_server_grab.py 456 {color_reset}")

def get_current_location() -> str:
    """
    Perform GPS lookup and return the current location as a string.
    """
    print(f"\n--- getting_current_location()")
    return f"<Current GPS location>: 80 Pasir Panjang"

def grab_transport(
    pickup_location: str,
    destination: str,
    service_type: str,
    schedule_time: str,
) -> str:
    """Book a Grab transport service.
    
    Args:
        destination: Where to drop off the passenger. If not specified, prompt user to input destination
        pickup_location: Where to pick up the passenger. MCP default: "<Current GPS location>"
        service_type: Type of service (GrabTaxi, GrabHitch, JustGrab). MCP default: "JustGrab"
        schedule_time: Schedule time for the pickup. MCP default: "<Current Time>"
        
    Returns:
        JSON string with booking details
        
    Note:
        When using this function through MCP, if pickup_location or schedule_time are not provided,
        Use the default values:
        - pickup_location defaults to "<Current GPS location>"
        - schedule_time defaults to "<Current Time>"
    """
    print(f"\n--- {color_green}grab_transport({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    # --- Simulated booking logic ---
    booking_id = f"GRB{random.randint(100000, 999999)}"

    # Resolve pickup point (use GPS fallback if none provided)
    if pickup_location.strip() == '':
        pickup_point = get_current_location()
    else:
        pickup_point = pickup_location.strip()

    # Determine whether this is a scheduled pickup
    is_scheduled = bool(schedule_time != "<Current Time>")

    # Force supported service types for scheduled pickups
    if is_scheduled:
        if service_type not in ["GrabHitch", "GrabShare"]:
            service_type = "GrabHitch"
        elif service_type == "JustGrab":
            service_type = "JustGrab (Advanced Booking)"
        timing = schedule_time
    else:
        timing = datetime.now().isoformat(timespec="seconds")

    # Fare calculation (extended multiplier list)
    fare_multipliers = {
        "GrabTaxi": 1.0,
        "GrabHitch": 0.7,
        "JustGrab": 0.8,
        "JustGrab (Advanced Booking)": 0.8,
        "GrabShare": 0.6,
    }
    estimated_fare = int(random.randint(8, 50) * fare_multipliers.get(service_type, 1.0))

    result = {
        "platform": "Grab",
        "Pickup Point": pickup_point,
        "Destination": destination,
        "Timing": timing,
        "Transport Type": service_type,
        "Scheduled": is_scheduled,
    }
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(f"{color_green}result_str = {result_str}{color_reset}")

    return result_str

def grab_food(restaurant: str, items: str, delivery_address: str) -> str:
    """Order food through GrabFood.
    
    Args:
        restaurant: Name of the restaurant
        items: Food items to order (comma-separated)
        delivery_address: Address for food delivery
        
    Returns:
        JSON string with order details
    """
    print(f"\n--- {color_green}grab_food({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    # Simulate food ordering logic
    order_id = f"GF{random.randint(100000, 999999)}"
    item_list = [item.strip() for item in items.split(',') if item.strip()]
    
    # Simulate pricing
    base_prices = {
        "big mac": 8.50,
        "mcchicken": 6.50,
        "fries": 3.50,
        "coke": 2.50,
        "nuggets": 7.00,
        "burger": 8.00,
        "pizza": 15.00,
        "nasi lemak": 5.50,
        "chicken rice": 4.50
    }
    
    total_price = 0
    order_items = []
    
    for item in item_list:
        item_lower = item.lower()
        # Find matching item or use default price
        price = next((price for key, price in base_prices.items() if key in item_lower), 8.00)
        total_price += price
        order_items.append({"item": item, "price": f"${price:.2f}"})
    
    delivery_fee = 2.50
    total_with_delivery = total_price + delivery_fee
    estimated_delivery = random.randint(25, 45)
    
    # Prepare structured output based on provided schema
    result = {
        "platform": "Grab",
        "food items": item_list,
        "sort_by": "Recommended",  # default value
        "restrictions": [],         # default empty list
        "delivery_mode": "Delivery",  # default value
        "cuisine_type": [],        # default empty list
    }
    result_str = json.dumps(result, indent=2)
    print(f"{color_green}result_str = {result_str}{color_reset}")
    return result_str

def grab_grocery(items: str, delivery_address: str, store_preference: str = "Any") -> str:
    """Order groceries through GrabMart.
    
    Args:
        items: Grocery items to order (comma-separated)
        delivery_address: Address for grocery delivery
        store_preference: Preferred store (Any, FairPrice, Cold Storage, Giant). MCP default: "Any"
        
    Returns:
        JSON string with grocery order details
    """
    print(f"\n--- {color_green}grab_grocery({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    
    # Simulate grocery ordering logic
    order_id = f"GM{random.randint(100000, 999999)}"
    item_list = [item.strip() for item in items.split(',') if item.strip()]
    
    # Simulate grocery pricing
    grocery_prices = {
        "milk": 3.50,
        "bread": 2.80,
        "eggs": 4.20,
        "rice": 8.90,
        "chicken": 12.50,
        "vegetables": 5.60,
        "fruits": 6.80,
        "yogurt": 4.50,
        "cheese": 7.20,
        "butter": 5.40,
        "oil": 6.30,
        "pasta": 3.90,
        "cereal": 8.50,
        "juice": 4.80,
        "detergent": 9.20,
        "toilet paper": 12.80,
        "shampoo": 8.90,
        "soap": 3.60
    }
    
    total_price = 0
    order_items = []
    
    for item in item_list:
        item_lower = item.lower()
        # Find matching item or use default price
        price = next((price for key, price in grocery_prices.items() if key in item_lower), 5.00)
        total_price += price
        order_items.append({"item": item, "price": f"${price:.2f}"})
    
    # Determine store based on preference
    available_stores = ["FairPrice", "Cold Storage", "Giant", "Sheng Siong"]
    if store_preference == "Any":
        selected_store = random.choice(available_stores)
    else:
        selected_store = store_preference if store_preference in available_stores else "FairPrice"
    
    delivery_fee = 3.99
    total_with_delivery = total_price + delivery_fee
    estimated_delivery = random.randint(60, 120)  # Groceries take longer
    
    # Prepare structured output
    result = {
        "platform": "Grab",
        "service": "GrabMart",
        "grocery_items": item_list,
        "store": selected_store,
        "delivery_address": delivery_address,
        "estimated_delivery_minutes": estimated_delivery,
        "total_price": f"${total_with_delivery:.2f}",
        "delivery_fee": f"${delivery_fee:.2f}"
    }
    result_str = json.dumps(result, indent=2)
    print(f"{color_green}result_str = {result_str}{color_reset}")
    return result_str

def do_nothing() -> str:
    """Do nothing, just return empty string.
        
    Returns:
        Empty string
        
    Note:
        When none of the features are available, use this formatting instead.
    """
    print(f"\n--- {color_green}do_nothing({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    return ""

def main(port: int = 7860):
    # Create separate interfaces for each function
    transport_demo = gr.Interface(
        fn=grab_transport,
        inputs=[
            gr.Textbox(label="Pickup Location", placeholder="e.g., Orchard Road (leave blank for current location)"),
            gr.Textbox(label="Destination", placeholder="e.g., Marina Bay Sands"),
            gr.Dropdown(choices=["GrabTaxi", "GrabHitch", "JustGrab"], label="Service Type", value="JustGrab"),
            gr.Textbox(label="Schedule Time (YYYY-MM-DD HH:MM) – optional", placeholder="e.g., 2024-12-31 18:30")
        ],
        outputs=gr.JSON(label="Transport Booking Details"),
        title="Grab Transport",
        description="Book a Grab transport service"
    )

    food_demo = gr.Interface(
        fn=grab_food,
        inputs=[
            gr.Textbox(label="Restaurant", placeholder="e.g., McDonald's"),
            gr.Textbox(label="Items", placeholder="e.g., Big Mac, Fries, Coke"),
            gr.Textbox(label="Delivery Address", placeholder="e.g., 123 Main Street")
        ],
        outputs=gr.JSON(label="Food Order Details"),
        title="Grab Food",
        description="Order food through GrabFood"
    )

    grocery_demo = gr.Interface(
        fn=grab_grocery,
        inputs=[
            gr.Textbox(label="Grocery Items", placeholder="e.g., milk, bread, eggs, rice"),
            gr.Textbox(label="Delivery Address", placeholder="e.g., 123 Main Street"),
            gr.Dropdown(choices=["Any", "FairPrice", "Cold Storage", "Giant", "Sheng Siong"], 
                       label="Store Preference", value="Any")
        ],
        outputs=gr.JSON(label="Grocery Order Details"),
        title="Grab Grocery (GrabMart)",
        description="Order groceries through GrabMart"
    )

    do_nothing_demo = gr.Interface(
        fn=do_nothing,
        inputs=[],
        outputs=gr.JSON(),
        title='',
        description='',
    )

    # Combine all interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [
            transport_demo, 
            food_demo, 
            grocery_demo,
            do_nothing_demo,
        ],
        [
            "Transport", 
            "Food", 
            "Grocery",
            "Do Nothing",
        ],
        title="Grab Services"
    )

    demo.launch(
        mcp_server=True,
        server_port=port,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(args.port)

"""
clear; python MCP_server_grab.py --port 7860
"""
