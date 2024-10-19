# ocr_llm.py

import logging
import json
import re
from apple_ocr.ocr import OCR
import ollama

# --------------------- Configuration ---------------------

SYSTEM_PROMPT = """You are a product recognition assistant. Your goal is to classify and extract the following details from product text:
1. Brand name
2. Expiry date
3. Package size
Provide the output strictly in JSON format with keys 'brand_name', 'expiry_date', and 'package_size'. Do not include any additional text, explanations, or formatting. If any of these details are missing, set their values to "Unknown". Package size can be in any format like net weight 1kg, 500g, 2L, etc."""

OLLAMA_MODEL = "qwen2.5:1.5b-instruct"  # Replace with your exact model name if different

# --------------------- Utility Functions ---------------------

def call_ollama(model: str, prompt: str) -> str:
    """
    Call the Ollama model with the given prompt.

    Args:
        model (str): The name of the Ollama model to use.
        prompt (str): The prompt to send to the model.

    Returns:
        str: The model's response.
    """
    try:
        # Construct messages as per Ollama's chat API
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ]
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except AttributeError as ae:
        logging.error(f"AttributeError when calling Ollama model: {ae}")
        return ""
    except Exception as e:
        logging.error(f"Error calling Ollama model: {e}")
        return ""

def clean_ollama_response(response: str) -> str:
    """
    Clean the Ollama response by removing code blocks if present.

    Args:
        response (str): The raw response from Ollama.

    Returns:
        str: The cleaned response.
    """
    # Remove markdown code blocks (```json ... ``` or ``` ...)
    response = re.sub(r'```json', '', response, flags=re.IGNORECASE)
    response = re.sub(r'```', '', response)
    response = response.strip()
    return response

def extract_product_details(images: list) -> dict:
    """
    Extract product details from multiple images using OCR and Ollama.

    Args:
        images (list of PIL.Image.Image): The images containing the product (front and back).

    Returns:
        dict: A dictionary with keys 'brand_name', 'expiry_date', and 'package_size'.
    """
    try:
        detected_texts = []

        # Perform OCR on each image
        for idx, image in enumerate(images):
            ocr_instance = OCR(image=image)
            ocr_result = ocr_instance.recognize()

            # Check the type of ocr_result
            if isinstance(ocr_result, dict):
                # Assuming the dictionary has a 'Content' key with a list of detected texts
                detected_text_list = ocr_result.get('Content', [])
                detected_text = " ".join(detected_text_list)
            elif hasattr(ocr_result, 'columns'):
                # Assuming it's a DataFrame
                if 'Content' in ocr_result.columns and not ocr_result.empty:
                    detected_text = " ".join(ocr_result['Content'].tolist())
                else:
                    detected_text = ""
            else:
                logging.error(f"Unexpected OCR result type: {type(ocr_result)}")
                detected_text = ""

            logging.info(f"OCR Detected Text from Image {idx + 1}: {detected_text}")
            if detected_text.strip() != "":
                detected_texts.append(detected_text)

        if not detected_texts:
            logging.warning("No product details found in any of the images.")
            return {"error": "No product details found"}

        # Combine all detected texts
        combined_text = " ".join(detected_texts)
        logging.info(f"Combined OCR Detected Text: {combined_text}")

        # Create the prompt for Ollama
        prompt = f"{combined_text}"

        # Call Ollama model
        ollama_response = call_ollama(model=OLLAMA_MODEL, prompt=prompt)
        logging.info(f"Ollama Raw Response: {ollama_response}")

        if not ollama_response:
            return {"error": "No response from Ollama model"}

        # Clean the response to remove any code blocks
        cleaned_response = clean_ollama_response(ollama_response)
        logging.info(f"Ollama Cleaned Response: {cleaned_response}")

        # Attempt to parse the response as JSON
        try:
            product_info = json.loads(cleaned_response)
            # Ensure all required keys are present and replace null with "Unknown"
            product_info = {
                "brand_name": product_info.get("brand_name") if product_info.get("brand_name") else "Unknown",
                "expiry_date": product_info.get("expiry_date") if product_info.get("expiry_date") else "Unknown",
                "package_size": product_info.get("package_size") if product_info.get("package_size") else "Unknown"
            }
            logging.info(f"Extracted Product Info: {product_info}")
            return product_info
        except json.JSONDecodeError:
            logging.error("Failed to parse Ollama response as JSON.")
            return {"error": "Failed to parse product details"}

    except Exception as e:
        logging.error(f"Error in extract_product_details: {e}")
        return {"error": "Error extracting product details"}
