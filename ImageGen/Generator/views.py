from django.shortcuts import render

from django.shortcuts import render, redirect
from django.utils.dateparse import parse_date

from datetime import datetime

from django.contrib.auth import authenticate, login, logout

from django.contrib import messages
from django.conf import settings

from PIL import Image
from typing import Union, Tuple
import io
import os
import base64
import boto3

import json

import matplotlib.pyplot as plt
import numpy as np

import sagemaker
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.__version__
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk_sagemaker.models import get_model_package_arn
from stability_sdk.api import GenerationRequest, GenerationResponse, TextPrompt



def home(request):

    return render(request, 'home.html', {})

def display_image(image, title):
    plt.figure(figsize=(12, 12))
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.title(title)
    plt.show()

def decode_and_show(model_response: GenerationResponse) -> None:
    """
    Decodes and displays an image from SDXL output

    Args:
        model_response (GenerationResponse): The response object from the deployed SDXL model.

    Returns:
        None
    """
    image = model_response.artifacts[0].base64
    image_data = base64.b64decode(image.encode())
    image = Image.open(io.BytesIO(image_data))
    print(image)
    image.show()
    return image

def getImage(request):

    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        '''endpoint_name ='Endpoint-Stable-Diffusion-XL-1-0-1'
        sagemaker_session = sagemaker.Session()
        deployed_model = StabilityPredictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
        output = deployed_model.predict(GenerationRequest(text_prompts=[TextPrompt(text=prompt)],
                                                # style_preset="cinematic",
                                                #seed = 12345,
                                                width=768,
                                                height=768.
                                                ))'''

        #image = decode_and_show(output)
        print("image_data")
        import requests

        API_URL = "https://onwuz8rcbwk1catz.us-east-1.aws.endpoints.huggingface.cloud"
        #headers = {"Authorization": "Bearer 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"}
        headers = {
	"Accept" : "image/png",
	"Authorization": "Bearer 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",
	"Content-Type": "application/json" 
      }

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        image_bytes = query({
            "inputs": prompt, "guidance_scale":10.5,})
        # You can access the image with PIL.Image for example
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image_bytes))
        image.show()
        
        
       # image_data = base64.b64encode(image).decode('utf-8')
        image_path = os.path.join(settings.BASE_DIR, 'static','images', 'lady.png')
        
        print(settings.BASE_DIR)
        print(image_path)
        print(image)
        image.save(image_path)

        return render(request, 'image.html')
