#IMPORTING REQUIRED LIBRARIES
!sudo apt install tesseract-ocr -y
!pip install pytesseract transformers

#UPLOADING FILE
from google.colab import files
from PIL import Image
uploaded = files.upload()
image_path = next(iter(uploaded))
image = Image.open(image_path)
image.show()

#PREPROCESSING AND EXTRACTING TEXT
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.title("Preprocessed Image")
plt.show()
text = pytesseract.image_to_string(thresh)
print("Extracted Text:\n", text)

#CLEANING TEXT
import re
def clean_text(text):

    text = text.encode("ascii", "ignore").decode()


    text = re.sub(r"\s+", " ", text)


    text = text.replace("Eopatity", "Equality")
    text = text.replace("tt", "it")
    text = text.replace("ts", "is")



    text = re.sub(r"[=–•…]", "", text)

    return text.strip()
cleaned_text = clean_text(text)
print("Cleaned Text:\n", cleaned_text)

#SUMMARIZING TEXT
from transformers import pipeline
summarizer = pipeline("summarization",model="facebook/bart-large-cnn")

if text.strip():
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    print("\n Summary:\n", summary[0]['summary_text'])
else:
    print("No readable text found in the image.")