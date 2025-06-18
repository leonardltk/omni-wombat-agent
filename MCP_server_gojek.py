import gradio as gr
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import random
import argparse
import pdb

color_cyan = "\033[96m"
color_reset = "\033[0m"

def get_current_location() -> str:
    """
    Perform GPS lookup and return the current location as a string.
    """
    print(f"\n--- getting_current_location()")
    return f"<Current GPS location>: 80 Pasir Panjang"

def gojek_transport(
    destination: str,
    pickup_location: str,
    service_type: str,
    schedule_time: str,
) -> str:
    """Book a Gojek transport service.
    
    Args:
        destination: Where to drop off the passenger. If not specified, prompt user to input destination
        pickup_location: Where to pick up the passenger. MCP default: "<Current GPS location>"
        service_type: Type of service (GoRide, GoCar, GoBluebird). MCP default: "GoRide"
        schedule_time: Schedule time for the pickup. MCP default: "<Current Time>"
        
    Returns:
        JSON string with booking details
        
    Note:
        When using this function through MCP, if pickup_location or schedule_time are not provided,
        they will automatically use the MCP default values:
        - pickup_location defaults to "<Current GPS location>"
        - schedule_time defaults to "<Current Time>"
    """
    print(f"\n--- {color_cyan}gojek_transport({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    # --- Simulated booking logic ---
    booking_id = f"GJK{random.randint(100000, 999999)}"

    # Resolve pickup point (use GPS fallback if none provided)
    if pickup_location.strip() == '':
        pickup_point = get_current_location()
    else:
        pickup_point = pickup_location.strip()

    # Determine whether this is a scheduled pickup
    is_scheduled = bool(schedule_time != "<Current Time>")

    # Force supported service types for scheduled pickups
    if is_scheduled:
        if service_type not in ["GoRide", "GoCar", "GoBluebird"]:
            service_type = "GoRide"
        timing = schedule_time
    else:
        timing = datetime.now().isoformat(timespec="seconds")

    # Fare calculation (extended multiplier list)
    fare_multipliers = {
        "GoRide": 1.0,
        "GoCar": 0.7,
        "GoBluebird": 0.8
    }
    estimated_fare = int(random.randint(8, 50) * fare_multipliers.get(service_type, 1.0))

    result = {
        "platform": "Gojek",
        "Pickup Point": pickup_point,
        "Destination": destination,
        "Timing": timing,
        "Transport Type": service_type,
        "Scheduled": is_scheduled,
    }
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(f"{color_cyan}result_str = {result_str}{color_reset}")

    return result_str

def gojek_food(restaurant: str, items: str, delivery_address: str) -> str:
    """Order food through GoFood.
    
    Args:
        restaurant: Name of the restaurant
        items: Food items to order (comma-separated)
        delivery_address: Address for food delivery
        
    Returns:
        JSON string with order details
    """
    print(f"\n--- {color_cyan}gojek_food({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    # Simulate food ordering logic
    order_id = f"GF{random.randint(100000, 999999)}"
    item_list = [item.strip() for item in items.split(',')]
    
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
    
    result = {
        "platform": "Gojek",
        "food items": item_list,
        "sort_by": "Recommended",  # default value
        "restrictions": [],         # default empty list
        "delivery_mode": "Delivery",  # default value
        "cuisine_type": [],        # default empty list
    }
    result_str = json.dumps(result, indent=2)
    print(f"{color_cyan}result_str = {result_str}{color_reset}")
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

def main(port: int = 7862):
    # Create separate interfaces for each function
    transport_demo = gr.Interface(
        fn=gojek_transport,
        inputs=[
            gr.Textbox(label="Pickup Location", placeholder="e.g., Orchard Road"),
            gr.Textbox(label="Destination", placeholder="e.g., Marina Bay Sands"),
            gr.Dropdown(choices=["GoCar", "GoTaxi"], label="Service Type", value="GoCar"),
            gr.Textbox(label="Schedule Time (YYYY-MM-DD HH:MM) – optional", placeholder="e.g., 2024-12-31 18:30")
        ],
        outputs=gr.JSON(label="Transport Booking Details"),
        title="Gojek Transport",
        description="Book a Gojek transport service"
    )

    food_demo = gr.Interface(
        fn=gojek_food,
        inputs=[
            gr.Textbox(label="Restaurant", placeholder="e.g., McDonald's"),
            gr.Textbox(label="Items", placeholder="e.g., Big Mac, Fries, Coke"),
            gr.Textbox(label="Delivery Address", placeholder="e.g., 123 Main Street")
        ],
        outputs=gr.JSON(label="Food Order Details"),
        title="GoFood",
        description="Order food through GoFood"
    )

    do_nothing_demo = gr.Interface(
        fn=do_nothing,
        inputs=[],
        outputs=gr.JSON(),
        title='',
        description='',
    )

    # Combine both interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [
            transport_demo,
            # food_demo,
            do_nothing_demo,
        ],
        [
            "Transport", 
            # "Food",
            "Do Nothing",
        ],
        title="Gojek Services"
    )

    demo.launch(
        mcp_server=True,
        server_port=port,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    main(args.port)

"""
clear; python MCP_server_gojek.py --port 7862
"""
