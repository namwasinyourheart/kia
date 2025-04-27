import logging
from datetime import date
from langchain.tools import tool
import json, math

@tool
def get_today(fmt: str = "%Y-%m-%d") -> str:
    """
    Return today's date as a string in the given format.

    Args:
        fmt: A date format string compatible with datetime.strftime.

    Returns:
        Formatted date string for today's date.
    """
    print(f"[Tool:get_today] Invoked with fmt={fmt}")
    return date.today().strftime(fmt)

@tool
def add(tool_input: str) -> str:
    """Add two numbers. Input as JSON: {"a": number, "b": number}."""
    data = json.loads(tool_input)
    a = data.get("a", 0)
    b = data.get("b", 0)
    print(f"[Tool:add] Invoked with a={a}, b={b}")
    return str(a + b)

@tool
def subtract(tool_input: str) -> str:
    """Subtract two numbers. Input as JSON: {"a": number, "b": number}. Returns a - b."""
    data = json.loads(tool_input)
    a = data.get("a", 0)
    b = data.get("b", 0)
    print(f"[Tool:subtract] Invoked with a={a}, b={b}")
    return str(a - b)

@tool
def multiply(tool_input: str) -> str:
    """Multiply two numbers. Input as JSON: {"a": number, "b": number}."""
    data = json.loads(tool_input)
    a = data.get("a", 0)
    b = data.get("b", 0)
    print(f"[Tool:multiply] Invoked with a={a}, b={b}")
    return str(a * b)

@tool
def divide(tool_input: str) -> str:
    """Divide two numbers. Input as JSON: {"a": number, "b": number}. Returns a / b."""
    data = json.loads(tool_input)
    a = data.get("a", 0)
    b = data.get("b", 1)
    print(f"[Tool:divide] Invoked with a={a}, b={b}")
    return str(a / b if b else "Infinity")

@tool
def sqrt(tool_input: str) -> str:
    """Square root. Input as JSON: {"x": number}."""
    data = json.loads(tool_input)
    x = data.get("x", 0)
    print(f"[Tool:sqrt] Invoked with x={x}")
    return str(math.sqrt(x))

@tool
def factorial(tool_input: str) -> str:
    """Factorial. Input as JSON: {"n": integer}."""
    data = json.loads(tool_input)
    n = int(data.get("n", 0))
    print(f"[Tool:factorial] Invoked with n={n}")
    return str(math.factorial(n))
