import gradio as gr
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import random
import argparse
import pdb

def gojek_transport(pickup_location: str, destination: str, service_type: str = "GoRide") -> str:
    """Book a Gojek transport service.
    
    Args:
        pickup_location: Where to pick up the passenger
        destination: Where to drop off the passenger
        service_type: Type of service (GoRide, GoCar, GoBluebird)
        
    Returns:
        JSON string with booking details
    """
    # Simulate booking logic
    booking_id = f"GJK{random.randint(100000, 999999)}"
    estimated_time = random.randint(5, 20)
    estimated_fare = random.randint(8, 50)
    
    # Adjust fare based on service type
    fare_multipliers = {
        "GoRide": 1.0,
        "GoCar": 0.7,
        "GoBluebird": 0.8
    }
    estimated_fare = int(estimated_fare * fare_multipliers.get(service_type, 1.0))
    
    result = {
        "platform": "Gojek",
        "booking_id": booking_id,
        "service_type": service_type,
        "pickup_location": pickup_location,
        "destination": destination,
        "estimated_arrival_time": f"{estimated_time} minutes",
        "estimated_fare": f"${estimated_fare}",
        "status": "confirmed",
        "driver_name": f"Driver {random.choice(['Mr A', 'Mr B', 'Mr C'])}"
    }
    
    return json.dumps(result, indent=2)

def gojek_food(restaurant: str, items: str, delivery_address: str) -> str:
    """Order food through GoFood.
    
    Args:
        restaurant: Name of the restaurant
        items: Food items to order (comma-separated)
        delivery_address: Address for food delivery
        
    Returns:
        JSON string with order details
    """
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
        "order_id": order_id,
        "restaurant": restaurant,
        "items": order_items,
        "subtotal": f"${total_price:.2f}",
        "delivery_fee": f"${delivery_fee:.2f}",
        "total": f"${total_with_delivery:.2f}",
        "delivery_address": delivery_address,
        "estimated_delivery_time": f"{estimated_delivery} minutes",
        "status": "confirmed",
        "delivery_partner": f"Rider {random.choice(['Ali', 'John', 'Priya', 'Chen'])}"
    }
    
    return json.dumps(result, indent=2)

def main(port: int = 7862):
    # Create separate interfaces for each function
    transport_demo = gr.Interface(
        fn=gojek_transport,
        inputs=[
            gr.Textbox(label="Pickup Location", placeholder="e.g., Orchard Road"),
            gr.Textbox(label="Destination", placeholder="e.g., Marina Bay Sands"),
            gr.Dropdown(choices=["GoRide", "GoCar", "GoBluebird"], label="Service Type", value="GoRide")
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

    # Combine both interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [transport_demo, food_demo],
        ["Transport", "Food"],
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
