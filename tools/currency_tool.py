"""
tools/currency_tool.py
Exchange Rate API tool — free tier, no key required.
MCP-pattern: schema-defined inputs, retry logic, graceful failure handling.
"""
import re
import time
import logging
import requests
from langchain_core.tools import tool

from core.prompts import CURRENCY_TOOL_DESCRIPTION
from config import CURRENCY_BASE_URL, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF

logger = logging.getLogger(__name__)

CURRENCY_NAMES = {
    "USD": "US Dollar", "INR": "Indian Rupee", "EUR": "Euro",
    "GBP": "British Pound", "JPY": "Japanese Yen", "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar", "CHF": "Swiss Franc", "CNY": "Chinese Yuan",
    "SGD": "Singapore Dollar", "AED": "UAE Dirham",
}


def _fetch_rate(base: str, target: str) -> dict:
    """
    Fetch exchange rate with exponential backoff retry.
    Uses exchangerate-api.com free tier — no API key needed.
    """
    url = f"{CURRENCY_BASE_URL}/{base.upper()}"
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                rates = data.get("rates", {})
                target_upper = target.upper()

                if target_upper not in rates:
                    return {
                        "success": False,
                        "error": f"Currency code '{target}' not found. Use standard 3-letter codes (e.g. USD, INR, EUR).",
                    }

                return {
                    "success": True,
                    "base": base.upper(),
                    "target": target_upper,
                    "rate": rates[target_upper],
                    "date": data.get("date", "latest"),
                }

            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": f"Currency code '{base}' not found. Use standard 3-letter codes.",
                }

            else:
                last_error = f"HTTP {response.status_code}"
                logger.warning("Currency API: %s (attempt %d)", last_error, attempt + 1)
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except requests.Timeout:
            last_error = "Request timed out."
            logger.warning("Currency API timeout (attempt %d).", attempt + 1)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except requests.ConnectionError:
            last_error = "Could not connect to exchange rate service."
            logger.warning("Currency API connection error (attempt %d).", attempt + 1)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except Exception:
            logger.exception("Unexpected error fetching rate %s→%s.", base, target)
            last_error = "Unexpected error."
            break

    return {
        "success": False,
        "error": f"Exchange rate data is currently unavailable. ({last_error})",
    }


def _parse_currency_input(query: str):
    """
    Parse currency codes from the agent's input.
    Handles: 'USD INR', 'USD to INR', 'USD,INR', 'convert USD to INR'.
    Returns (base, target) or (None, None) if parsing fails.
    """
    codes = re.findall(r'\b[A-Za-z]{3}\b', query.upper())
    codes = [c for c in codes if len(c) == 3]

    if len(codes) >= 2:
        return codes[0], codes[1]
    elif len(codes) == 1:
        return codes[0], "INR"
    return None, None


def create_currency_tool():
    """Factory that creates the currency tool."""

    def currency_tool(query: str) -> str:
        base, target = _parse_currency_input(query)

        if not base:
            return (
                "Please specify the currencies you want to convert. "
                "Example: 'USD to INR' or 'EUR GBP'"
            )

        result = _fetch_rate(base, target)

        if not result["success"]:
            return f"❌ {result['error']}"

        rate        = result["rate"]
        base_name   = CURRENCY_NAMES.get(base,   base)
        target_name = CURRENCY_NAMES.get(target, target)

        return (
            f"💱 Exchange Rate ({result['date']}):\n"
            f"  1 {base} ({base_name}) = {rate:,.4f} {target} ({target_name})\n"
            f"  Example: 1000 {base} = {rate * 1000:,.2f} {target}\n"
            f"  [Live data from ExchangeRate-API]"
        )

    currency_tool.__doc__ = CURRENCY_TOOL_DESCRIPTION
    return tool(currency_tool)
