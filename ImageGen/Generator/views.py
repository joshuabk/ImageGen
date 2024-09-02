from django.shortcuts import render

from django.shortcuts import render, redirect
from django.utils.dateparse import parse_date

from datetime import datetime

from django.contrib.auth import authenticate, login, logout

from django.contrib import messages

from PIL import Image
from typing import Union, Tuple
import io
import os
import base64
import boto3

import json

import matplotlib.pyplot as plt
import numpy as np
from sagemaker.jumpstart.model import JumpStartMode
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
    image.show()

def GetImage(prompt):
    endpoint_name ='Endpoint-Stable-Diffusion-XL-1-0-1'
    sagemaker_session = sagemaker.Session()
    deployed_model = StabilityPredictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
    output = deployed_model.predict(GenerationRequest(text_prompts=[TextPrompt(text="pixar woman in a space ship")],
                                             # style_preset="cinematic",
                                             #seed = 12345,
                                            width=1024,
                                            height=1024.
                                             ))

    decode_and_show(output)
