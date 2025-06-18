import gradio as gr
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import random
import argparse

color_red = "\033[91m"
color_reset = "\033[0m"

def redmart_grocery(items: str, delivery_address: str, delivery_slot: str = "Next Available", membership_tier: str = "Standard") -> str:
    """Order groceries through RedMart.
    
    Args:
        items: Grocery items to order (comma-separated)
        delivery_address: Address for grocery delivery
        delivery_slot: Preferred delivery slot (Next Available, Morning, Afternoon, Evening). MCP default: "Next Available"
        membership_tier: Membership tier (Standard, Premium, VIP). MCP default: "Standard"
        
    Returns:
        JSON string with grocery order details
    """
    print(f"\n--- {color_red}redmart_grocery({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    
    # Simulate grocery ordering logic
    order_id = f"RM{random.randint(100000, 999999)}"
    item_list = [item.strip() for item in items.split(',') if item.strip()]
    
    # RedMart specific grocery pricing (slightly different from Grab)
    grocery_prices = {
        "milk": 3.20,
        "bread": 2.50,
        "eggs": 3.90,
        "rice": 7.80,
        "chicken": 11.90,
        "vegetables": 4.80,
        "fruits": 6.20,
        "yogurt": 4.10,
        "cheese": 6.80,
        "butter": 5.10,
        "oil": 5.90,
        "pasta": 3.60,
        "cereal": 7.90,
        "juice": 4.40,
        "detergent": 8.50,
        "toilet paper": 11.90,
        "shampoo": 8.20,
        "soap": 3.20,
        "organic vegetables": 8.50,
        "premium meat": 18.90,
        "imported cheese": 12.50,
        "wine": 25.00,
        "coffee beans": 15.80
    }
    
    total_price = 0
    order_items = []
    
    for item in item_list:
        item_lower = item.lower()
        # Find matching item or use default price
        price = next((price for key, price in grocery_prices.items() if key in item_lower), 4.50)
        total_price += price
        order_items.append({"item": item, "price": f"${price:.2f}"})
    
    # Apply membership discounts
    discount_rates = {
        "Standard": 0.0,
        "Premium": 0.05,  # 5% discount
        "VIP": 0.10       # 10% discount
    }
    discount_rate = discount_rates.get(membership_tier, 0.0)
    discounted_price = total_price * (1 - discount_rate)
    
    # Delivery fee based on membership and order value
    if membership_tier == "VIP" or discounted_price > 50:
        delivery_fee = 0.0  # Free delivery
    elif membership_tier == "Premium":
        delivery_fee = 1.99
    else:
        delivery_fee = 3.99
    
    total_with_delivery = discounted_price + delivery_fee
    
    # Determine delivery time based on slot
    slot_times = {
        "Next Available": random.randint(45, 90),
        "Morning": random.randint(480, 720),    # 8-12 hours
        "Afternoon": random.randint(720, 960),  # 12-16 hours
        "Evening": random.randint(960, 1200)    # 16-20 hours
    }
    estimated_delivery = slot_times.get(delivery_slot, 60)
    
    # Prepare structured output
    result = {
        "platform": "RedMart",
        "service": "RedMart Grocery",
        "grocery_items": item_list,
        "delivery_address": delivery_address,
        "delivery_slot": delivery_slot,
        "membership_tier": membership_tier,
        "discount_applied": f"{discount_rate*100:.0f}%",
        "subtotal": f"${total_price:.2f}",
        "discounted_total": f"${discounted_price:.2f}",
        "delivery_fee": f"${delivery_fee:.2f}",
        "final_total": f"${total_with_delivery:.2f}",
        "estimated_delivery_minutes": estimated_delivery,
        "order_id": order_id
    }
    result_str = json.dumps(result, indent=2)
    print(f"{color_red}result_str = {result_str}{color_reset}")
    return result_str

def redmart_meal_kit(meal_type: str, servings: int, delivery_address: str, dietary_preferences: str = "None") -> str:
    """Order meal kits through RedMart.
    
    Args:
        meal_type: Type of meal kit (Asian, Western, Vegetarian, Healthy, Family)
        servings: Number of servings (1-6)
        delivery_address: Address for delivery
        dietary_preferences: Dietary restrictions (None, Halal, Vegetarian, Vegan, Gluten-Free). MCP default: "None"
        
    Returns:
        JSON string with meal kit order details
    """
    print(f"\n--- {color_red}redmart_meal_kit({json.dumps(locals(), indent=4, ensure_ascii=False)}{color_reset})")
    
    order_id = f"RMK{random.randint(100000, 999999)}"
    
    # Meal kit pricing
    base_prices = {
        "Asian": 12.90,
        "Western": 14.90,
        "Vegetarian": 11.90,
        "Healthy": 15.90,
        "Family": 18.90
    }
    
    base_price = base_prices.get(meal_type, 13.90)
    total_price = base_price * servings
    
    # Available meal kits based on type and dietary preferences
    meal_options = {
        "Asian": ["Chicken Teriyaki", "Beef Rendang", "Thai Green Curry", "Korean BBQ"],
        "Western": ["Grilled Salmon", "Pasta Carbonara", "Beef Steak", "Chicken Parmesan"],
        "Vegetarian": ["Mushroom Risotto", "Veggie Stir Fry", "Caprese Salad", "Quinoa Bowl"],
        "Healthy": ["Grilled Fish", "Quinoa Salad", "Steamed Vegetables", "Protein Bowl"],
        "Family": ["Family Pizza Kit", "Taco Night", "Pasta Family Pack", "BBQ Set"]
    }
    
    selected_meal = random.choice(meal_options.get(meal_type, ["Mixed Meal Kit"]))
    
    delivery_fee = 2.99 if total_price < 30 else 0.0
    final_total = total_price + delivery_fee
    estimated_delivery = random.randint(120, 180)  # Meal kits need more prep time
    
    result = {
        "platform": "RedMart",
        "service": "RedMart Meal Kit",
        "meal_type": meal_type,
        "selected_meal": selected_meal,
        "servings": servings,
        "dietary_preferences": dietary_preferences,
        "delivery_address": delivery_address,
        "price_per_serving": f"${base_price:.2f}",
        "subtotal": f"${total_price:.2f}",
        "delivery_fee": f"${delivery_fee:.2f}",
        "total": f"${final_total:.2f}",
        "estimated_delivery_minutes": estimated_delivery,
        "order_id": order_id
    }
    result_str = json.dumps(result, indent=2)
    print(f"{color_red}result_str = {result_str}{color_reset}")
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

def main(port: int = 7864):
    # Create separate interfaces for each function
    grocery_demo = gr.Interface(
        fn=redmart_grocery,
        inputs=[
            gr.Textbox(label="Grocery Items", placeholder="e.g., organic vegetables, premium meat, wine, coffee beans"),
            gr.Textbox(label="Delivery Address", placeholder="e.g., 123 Main Street"),
            gr.Dropdown(choices=["Next Available", "Morning", "Afternoon", "Evening"], 
                       label="Delivery Slot", value="Next Available"),
            gr.Dropdown(choices=["Standard", "Premium", "VIP"], 
                       label="Membership Tier", value="Standard")
        ],
        outputs=gr.JSON(label="Grocery Order Details"),
        title="RedMart Grocery",
        description="Order groceries through RedMart with membership benefits"
    )

    meal_kit_demo = gr.Interface(
        fn=redmart_meal_kit,
        inputs=[
            gr.Dropdown(choices=["Asian", "Western", "Vegetarian", "Healthy", "Family"], 
                       label="Meal Type", value="Asian"),
            gr.Slider(minimum=1, maximum=6, value=2, step=1, label="Number of Servings"),
            gr.Textbox(label="Delivery Address", placeholder="e.g., 123 Main Street"),
            gr.Dropdown(choices=["None", "Halal", "Vegetarian", "Vegan", "Gluten-Free"], 
                       label="Dietary Preferences", value="None")
        ],
        outputs=gr.JSON(label="Meal Kit Order Details"),
        title="RedMart Meal Kit",
        description="Order meal kits through RedMart"
    )

    do_nothing_demo = gr.Interface(
        fn=do_nothing,
        inputs=[],
        outputs=gr.JSON(),
        title='',
        description='',
    )

    # Combine interfaces in a tabbed interface
    demo = gr.TabbedInterface(
        [
            grocery_demo,
            # meal_kit_demo,
            do_nothing_demo,
        ],
        [
            "Grocery",
            # "Meal Kit",
            "Do Nothing",
        ],
        title="RedMart Services"
    )

    demo.launch(
        mcp_server=True,
        server_port=port,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()
    main(args.port)

"""
Usage:
clear; python MCP_server_RedMart.py --port 7864
""" 