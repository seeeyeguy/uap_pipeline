#main.py file for document ingestion pipeline

from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "THUDM/glm-ocr-0.9b"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device="auto",
    trust_remote_code=True
)

# TODO: Convert this to iterate through any file not already completed
# which would require marking files as complete somewhere
file_name = "291645977.pdf"
images = convert_from_path(pdf_path=f"./pdf/{file_name}", output_folder="./images/")

for i, image in enumerate(images):
    image.save(f'{file_name}.jpg', 'JPEG')






