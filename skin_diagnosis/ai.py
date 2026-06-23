import json
import logging
import time
from django.conf import settings
from google.generativeai import configure, GenerativeModel
from google.api_core.exceptions import GoogleAPIError, RetryError, ServiceUnavailable
import mimetypes  # Import mimetypes to guess the file type

logger = logging.getLogger(__name__)


MAX_RETRIES = 3
RETRY_DELAY = 1


def analyze_skin_image(image_path: str) -> dict:
    """
    Returns structured analysis:
    {
        "conditions": list[str],
        "confidence": float,
        "recommendations": list[str],
        "urgency": "low|medium|high",
        "raw_response": str
    }
    """
    try:
        api_key = getattr(settings, "GEMINI_API_KEY", None)
        if not api_key:
            raise ValueError("Missing Gemini API key in settings")

        configure(api_key=api_key)
        model = GenerativeModel("gemini-2.0-flash")

        prompt = """Analyze this skin condition image and provide:
        1. Top 3 possible conditions (array)
        2. Confidence score (0-1)
        3. 3 recommended actions (array)
        4. Urgency level (low/medium/high)

        Return ONLY valid JSON format:
        {
            "conditions": [],
            "confidence": 0.0,
            "recommendations": [],
            "urgency": ""
        }"""

        try:
            # Validate file existence
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Validate file size
            file_size = len(image_data)
            max_size = 10 * 1024 * 1024  # 10MB
            if file_size > max_size:
                logger.warning(f"Image too large: {file_size} bytes (max: {max_size})")
                return {"error": "Image file too large (max 10MB)"}

            # Validate file is not empty
            if file_size == 0:
                logger.error(f"Empty image file: {image_path}")
                return {"error": "Image file is empty"}

            # Basic image header validation
            # Check for common image file signatures
            is_valid_image = False
            signatures = {
                b"\xff\xd8\xff": "JPEG",
                b"\x89PNG\r\n\x1a\n": "PNG",
                b"GIF87a": "GIF",
                b"GIF89a": "GIF",
                b"RIFF": "WEBP",
            }

            for sig, format_name in signatures.items():
                if image_data.startswith(sig):
                    is_valid_image = True
                    logger.info(f"Detected image format: {format_name}")
                    break

            if not is_valid_image:
                logger.warning(
                    f"File does not appear to be a valid image: {image_path}"
                )
                return {"error": "File does not appear to be a valid image"}

        except FileNotFoundError:
            logger.error(f"Image file not found at {image_path}")
            return {"error": "Image file not found"}
        except IOError as e:
            logger.error(f"Could not read image file {image_path}: {e}")
            return {"error": "Could not read image file"}

        # Guess the MIME type of the image
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None or not mime_type.startswith("image/"):
            logger.warning(
                f"Could not determine image MIME type or it's not an image: {image_path}"
            )
            # Fallback to a common type or return error
            mime_type = "image/jpeg"  # Or handle as an error

        # Pass the prompt and image data with MIME type to the model
        image_part = {"mime_type": mime_type, "data": image_data}

        # Implement retry mechanism for transient errors
        response = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"API call attempt {attempt}/{MAX_RETRIES}")
                response = model.generate_content([prompt, image_part])
                break  # Success, exit the retry loop
            except (RetryError, ServiceUnavailable) as e:
                if attempt < MAX_RETRIES:
                    # Calculate exponential backoff delay: 1s, 2s, 4s, etc.
                    delay = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Transient API error: {str(e)}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"API call failed after {MAX_RETRIES} attempts: {str(e)}"
                    )
                    raise  # Re-raise the exception after all retries failed

        if not response:
            raise ValueError("Failed to get response from API after retries")

        try:
            # More robust JSON extraction using regex
            import re

            # First, try to extract JSON from code blocks
            json_match = re.search(
                r"```(?:json)?\s*({.*?})\s*```", response.text, re.DOTALL
            )

            if json_match:
                # Found JSON in code block
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON without code blocks - look for a pattern that looks like JSON
                json_match = re.search(r'({[\s\S]*"conditions"[\s\S]*})', response.text)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # Fallback to the entire response if no JSON pattern found
                    json_str = response.text

            # Parse the JSON
            result = json.loads(json_str)

            # Validate required fields
            required_fields = ["conditions", "confidence", "recommendations", "urgency"]
            missing_fields = [field for field in required_fields if field not in result]

            if missing_fields:
                logger.warning(f"Missing fields in response: {missing_fields}")
                # Add missing fields with default values
                for field in missing_fields:
                    if field == "conditions":
                        result[field] = []
                    elif field == "confidence":
                        result[field] = 0.0
                    elif field == "recommendations":
                        result[field] = ["Consult a dermatologist"]
                    elif field == "urgency":
                        result[field] = "medium"

            # Add the raw response for debugging
            result["raw_response"] = response.text
            return result
        except (IndexError, json.JSONDecodeError) as e:
            logger.warning(
                f"Failed to parse response: {e}. Raw response: {response.text}"
            )
            return {
                "error": "Analysis format error",
                "raw_response": response.text,
                "conditions": [],
                "confidence": 0.0,
                "recommendations": ["Consult a dermatologist"],
                "urgency": "medium",
            }
        except AttributeError:
            # Handle cases where response might not have a .text attribute
            logger.warning(
                f"Response object has no text attribute. Response: {response}"
            )
            return {
                "error": "Unexpected response format",
                "raw_response": str(response),  # Try to represent response as string
                "conditions": [],
                "confidence": 3.0,
                "recommendations": ["Consult a dermatologist"],
                "urgency": "medium",
            }

    except GoogleAPIError as e:
        logger.error(f"Gemini API error: {str(e)}")
        return {"error": "AI service unavailable"}
    except ValueError as e:  # Catch the specific ValueError for missing API key
        logger.error(f"Configuration error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Analysis failed"}
