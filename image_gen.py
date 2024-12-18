import replicate
from PIL import Image
import requests
from io import BytesIO
import time

def load_image_from_url(input_image: str) -> Image.Image:
    response = requests.get(input_image)
    response.raise_for_status()  # Raise an exception for HTTP errors
    image = Image.open(BytesIO(response.content))
    return image

def generate_image(prompt: str):
    start_time = time.time()
    print("Generating image")
    img_url = replicate.run("black-forest-labs/flux-schnell",
        input={
            "prompt": "3D icon of a " +prompt+", white background",
        })[0]
    print(f"Image generated in {time.time()-start_time}s")
    print("url: ", img_url)
    return load_image_from_url(img_url)