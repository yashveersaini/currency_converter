import os
import json
from typing import Annotated

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

import requests
from langchain_core.tools import tool, InjectedToolArg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Setup
load_dotenv()

EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)


# Tools 
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """
    Fetches the currency conversion factor between a given base currency
    and a target currency.
    """
    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}"
        f"/pair/{base_currency}/{target_currency}"
    )
    response = requests.get(url)
    return response.json()


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """
    Given a currency conversion rate, calculates the target currency value
    from a given base currency value.
    """
    return base_currency_value * conversion_rate


# LLM + tool binding
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

llm_with_tools = llm.bind_tools([convert, get_conversion_factor])


# Core conversion logic 
def run_conversion(amount: float, base_currency: str, target_currency: str):
    query = (
        f"What is the conversion factor between {base_currency} and "
        f"{target_currency}, and based on that can you convert {amount} "
        f"{base_currency} to {target_currency}. "
        f"Call ALL necessary tools in parallel at the same time."
    )

    messages = [HumanMessage(query)]

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    conversion_rate = None
    converted_value = None

    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            tool_message = get_conversion_factor.invoke(tool_call)
            conversion_rate = json.loads(tool_message.content)["conversion_rate"]
            messages.append(tool_message)

        if tool_call["name"] == "convert":
            tool_call["args"]["conversion_rate"] = conversion_rate
            tool_message = convert.invoke(tool_call)
            converted_value = tool_message.content
            messages.append(tool_message)

    final_response = llm_with_tools.invoke(messages)

    return {
        "conversion_rate": conversion_rate,
        "converted_value": converted_value,
        "summary": final_response.content,
    }


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/convert", methods=["POST"])
def api_convert():
    data = request.get_json()

    try:
        amount = float(data.get("amount"))
        base_currency = data.get("base_currency", "").upper().strip()
        target_currency = data.get("target_currency", "").upper().strip()

        if not base_currency or not target_currency:
            return jsonify({"error": "Both currencies are required."}), 400

        result = run_conversion(amount, base_currency, target_currency)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)