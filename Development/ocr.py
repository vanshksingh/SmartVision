from apple_ocr.ocr import OCR
from PIL import Image

# Load the image
image = Image.open("/Users/vanshkumarsingh/Desktop/test2.png")

# Initialize OCR instance
ocr_instance = OCR(image=image)

# Recognize text and get the structured DataFrame
dataframe = ocr_instance.recognize()

# Extract only the detected text column (assuming the column is named 'Content')
detected_text = dataframe['Content']

# Print the detected text
print(detected_text)
