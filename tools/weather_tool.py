"""
tools/weather_tool.py
OpenWeatherMap API tool — MCP-pattern implementation.
Schema-defined inputs, agent-driven selection, failure handling with retry.
"""
import time
import logging
import requests
from langchain_core.tools import tool

from core.prompts import WEATHER_TOOL_DESCRIPTION
from config import (
    WEATHER_API_KEY, WEATHER_BASE_URL,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
)

logger = logging.getLogger(__name__)


def _fetch_weather(city: str) -> dict:
    """
    Internal function — fetches weather data with exponential backoff retry.
    Separated from the tool so retry logic is reusable and independently testable.
    """
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                WEATHER_BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}

            elif response.status_code == 404:
                return {
                    "success": False,
                    "error": f"City '{city}' not found. Please check the city name and try again.",
                }

            elif response.status_code == 401:
                return {
                    "success": False,
                    "error": "Weather API authentication failed. Please check the API key.",
                }

            elif response.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt) * 2
                logger.warning("Weather API rate-limited. Waiting %.1fs before retry %d.", wait, attempt + 1)
                time.sleep(wait)
                last_error = "Weather API rate limit reached."
                continue

            else:
                last_error = f"Unexpected response (HTTP {response.status_code})"
                logger.warning("Weather API: %s (attempt %d)", last_error, attempt + 1)
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except requests.Timeout:
            last_error = "Request timed out."
            logger.warning("Weather API timeout (attempt %d).", attempt + 1)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except requests.ConnectionError:
            last_error = "Could not connect to the weather service."
            logger.warning("Weather API connection error (attempt %d).", attempt + 1)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))

        except Exception:
            logger.exception("Unexpected error fetching weather for '%s'.", city)
            last_error = "Unexpected error."
            break

    return {
        "success": False,
        "error": f"Live weather data is currently unavailable. ({last_error})",
    }


def _format_weather_response(data: dict, city: str) -> str:
    """Format raw OpenWeatherMap API response into readable output."""
    try:
        name        = data.get("name", city)
        country     = data.get("sys", {}).get("country", "")
        temp        = data["main"]["temp"]
        feels_like  = data["main"]["feels_like"]
        humidity    = data["main"]["humidity"]
        description = data["weather"][0]["description"].capitalize()
        wind_speed  = data["wind"]["speed"]
        visibility  = data.get("visibility", "N/A")
        if visibility != "N/A":
            visibility = f"{visibility / 1000:.1f} km"

        return (
            f"🌤 Weather in {name}, {country}:\n"
            f"  Condition    : {description}\n"
            f"  Temperature  : {temp}°C (feels like {feels_like}°C)\n"
            f"  Humidity     : {humidity}%\n"
            f"  Wind speed   : {wind_speed} m/s\n"
            f"  Visibility   : {visibility}\n"
            f"  [Live data from OpenWeatherMap API]"
        )
    except KeyError as exc:
        logger.error("Missing field in weather response: %s", exc)
        return f"Weather data received but could not be parsed. Missing field: {exc}"


def create_weather_tool():
    """Factory that creates the weather tool."""

    def weather_tool(city: str) -> str:
        if not WEATHER_API_KEY:
            return (
                "⚠️ Weather API key not configured. "
                "Set WEATHER_API_KEY in your .env file to enable live weather data."
            )

        city = city.strip().strip('"').strip("'")
        result = _fetch_weather(city)

        if not result["success"]:
            return f"❌ {result['error']}"

        return _format_weather_response(result["data"], city)

    weather_tool.__doc__ = WEATHER_TOOL_DESCRIPTION
    return tool(weather_tool)
