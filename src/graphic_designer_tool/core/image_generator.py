""" Image generator core class """

import io

import openai
import requests
from PIL import Image


class ImageGenerator:
    """
    A class to generate images using OpenAI's DALL-E model based on text descriptions.

    Attributes:
        model (str): The name of the AI model to use for image generation.
        quality (str): The desired quality level of the generated images.
        n (int): The number of images to generate.
        dim (str): The dimensions of the images. Must be one of "1024x1024",
            "1024x1792", or "1792x1024".

    Raises:
        ValueError: If `dim` is not one of the allowed dimension values.
    """

    def __init__(self, model: str, quality: str, n: int, dim: str):
        """
        Initializes the ImageGenerator with specified model parameters.

        Args:
            model (str): The AI model to use for image generation.
            quality (str): Quality level for generated images (e.g., "high", "medium", "low").
            n (int): Number of images to generate per request.
            dim (str): Dimensions of generated images. Allowed values are "1024x1024",
                "1024x1792", or "1792x1024".

        Raises:
            ValueError: If `dim` is not one of the accepted dimension formats.
        """
        self.model = model
        self.quality = quality
        self.n = n
        self.dim = dim

        if self.dim not in ["1024x1024", "1024x1792", "1792x1024"]:
            raise ValueError(
                "Dimension must be '1024x1024', '1024x1792' or '1792x1024'"
            )

    def generate_visual(self, visual_description: str) -> list[str]:
        """
        Generates image URLs based on a given description.

        Args:
            visual_description (str): A text prompt describing the desired visual content.

        Returns:
            list[str]: A list of URLs for the generated images.
        """
        responses = openai.images.generate(
            model=self.model,
            prompt=f"Create a visual following this description: {visual_description}",
            size=self.dim,
            quality=self.quality,
            n=self.n,
        ).data

        return [response.url for response in responses]

    @staticmethod
    def get_image_from_url(url: str) -> bytes:
        """
        Downloads and returns the raw image content from a specified URL.

        Args:
            url (str): The URL of the image to download.

        Returns:
            bytes: The raw byte content of the image, suitable for saving or displaying.
        """
        return requests.get(url, timeout=10).content

    def get_images(self, visual_description: str) -> Image:
        """
        Generates images from a description and retrieves their byte content.

        Args:
            visual_description (str): A text prompt describing the desired visual content.

        Returns:
            bytes: generated image.
        """
        generated_image_urls = self.generate_visual(
            visual_description=visual_description
        )

        images = []
        for image_url in generated_image_urls:
            images.append(self.get_image_from_url(image_url))

        return Image.open(io.BytesIO(images[0]))
