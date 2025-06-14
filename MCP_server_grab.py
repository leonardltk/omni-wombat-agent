import gradio as gr
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import random
import argparse

color_green = "\033[92m"
color_reset = "\033[0m"
print(f"{color_green}MCP_server_grab.py 456 {color_reset}")

def _simulate_gps_location() -> str:
    """Simulate a GPS lookup and return the current coordinates as 'longitude, latitude'."""
    print(f"\n--- _simulate_gps_location()")
    return f"<Current GPS location>"

def grab_transport(
    destination: str,
    pickup_location: str,
    service_type: str,
    schedule_time: str,
) -> str:
    """Book a Grab transport service.
    
    Args:
        destination: Where to drop off the passenger. If not specified, prompt user to input destination
        pickup_location: Where to pick up the passenger. If not specified, Default to "<Current GPS location>"
        service_type: Type of service (GrabTaxi, GrabHitch, JustGrab). Default is JustGrab
        schedule_time: Schedule time for the pickup. If not specified by user, default to "<Current Time>"
        
    Returns:
        JSON string with booking details
    """
    print(f"\n--- {color_green}grab_transport({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    # --- Simulated booking logic ---
    booking_id = f"GRB{random.randint(100000, 999999)}"

    # Resolve pickup point (use GPS fallback if none provided)
    if pickup_location.strip() == '':
        pickup_point = _simulate_gps_location()
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

    # Combine both interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [transport_demo, food_demo],
        ["Transport", "Food"],
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
