# main.py

import logging
import sys
import json
from PIL import Image

from config import CONFIG
from models import load_yolo_model, load_resnet_model, get_transform
from detection import real_time_detection
from ocr_llm import extract_product_details


def process_single_or_double_images(front_image_path, back_image_path=None):
    """
    Process one or two images (front and back) to extract product details.

    Args:
        front_image_path (str): Path to the front image.
        back_image_path (str, optional): Path to the back image.

    Returns:
        dict: Extracted product details or error information.
    """
    try:
        images = []

        # Load front image
        front_image = Image.open(front_image_path)
        logging.info(f"Loaded front image: {front_image_path}")
        images.append(front_image)

        # Load back image if provided
        if back_image_path:
            back_image = Image.open(back_image_path)
            logging.info(f"Loaded back image: {back_image_path}")
            images.append(back_image)

        # Extract product details
        product_details = extract_product_details(images)
        return product_details

    except Exception as e:
        logging.error(f"Error processing images: {e}")
        return {"error": "Error processing images"}


def main():
    """
    Main function to run real-time detection or process images based on user input.
    """
    try:
        if len(sys.argv) == 1:
            # No arguments provided, start real-time detection
            logging.info("Starting real-time detection...")

            # Load models
            yolo_model = load_yolo_model(CONFIG["YOLO_MODEL_PATH"])
            resnet_model = load_resnet_model(CONFIG["RESNET_MODEL_PATH"], CONFIG["CLASS_NAMES"])

            # Define the transform for the ResNet model
            transform = get_transform(CONFIG["OCR_IMAGE_SIZE"])

            # Start real-time object detection and classification
            real_time_detection(yolo_model, resnet_model, transform, CONFIG)

        elif len(sys.argv) == 2 or len(sys.argv) == 3:
            # Process single or double images
            front_image_path = sys.argv[1]
            back_image_path = sys.argv[2] if len(sys.argv) == 3 else None

            logging.info("Starting image processing...")

            product_details = process_single_or_double_images(front_image_path, back_image_path)

            # Print the structured product details
            print(json.dumps(product_details, indent=4))

        else:
            logging.error("Invalid number of arguments provided.")
            print("Usage for real-time detection:")
            print("    python main.py")
            print("Usage for processing images:")
            print("    python main.py <front_image_path> [back_image_path]")
            sys.exit(1)

    except Exception as e:
        logging.critical(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
