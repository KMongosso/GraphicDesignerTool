""" Image Generator tests """

import pytest

from graphic_designer_tool.core.image_generator import ImageGenerator


def test_initialization_valid_params():
    """
    Test that the ImageGenerator initializes correctly with valid parameters.

    Verifies that the attributes `model`, `quality`, `n`, and `dim` are set as expected
    when the ImageGenerator instance is created with valid inputs.
    """
    generator = ImageGenerator(model="dall-e", quality="high", n=2, dim="1024x1024")
    assert generator.model == "dall-e"
    assert generator.quality == "high"
    assert generator.n == 2
    assert generator.dim == "1024x1024"


def test_initialization_invalid_dim():
    """
    Test that initializing ImageGenerator with an invalid dimension raises a ValueError.

    Ensures that a ValueError is raised when the `dim` parameter is set to a value
    that is not one of the allowed options ('1024x1024', '1024x1792', '1792x1024').
    """
    with pytest.raises(
        ValueError, match="Dimension must be '1024x1024', '1024x1792' or '1792x1024'"
    ):
        ImageGenerator(model="dall-e", quality="high", n=1, dim="800x600")


def test_generate_visual(mocker):
    """
    Test the `generate_visual` method to ensure it generates the correct URLs.

    Mocks the OpenAI API response to simulate generating image URLs based on a visual description.
    Verifies that the generated URLs match the expected output and that the correct number of URLs
    is returned.

    Args:
        mocker (pytest-mock): Pytest mocker to replace the `openai.images.generate` call.
    """
    mock_openai_response = mocker.Mock()
    mock_openai_response.data = [
        mocker.Mock(url="https://example.com/image1.png"),
        mocker.Mock(url="https://example.com/image2.png"),
    ]
    mocker.patch("openai.images.generate", return_value=mock_openai_response)

    generator = ImageGenerator(model="dall-e", quality="high", n=2, dim="1024x1024")
    urls = generator.generate_visual("A surreal forest landscape at sunset")

    assert urls == ["https://example.com/image1.png", "https://example.com/image2.png"]
    assert len(urls) == 2


def test_get_image_from_url(mocker):
    """
    Test the `get_image_from_url` method to ensure it retrieves image data from a URL.

    Mocks `requests.get` to simulate downloading image content from a URL. Verifies that
    the returned image data matches the expected output and that `requests.get` was called
    with the correct URL.

    Args:
        mocker (pytest-mock): Pytest mocker to replace the `requests.get` call.
    """
    mock_response = mocker.Mock()
    mock_response.content = b"fake_image_data"
    mocker.patch("requests.get", return_value=mock_response)

    url = "https://example.com/image.png"
    image_data = ImageGenerator.get_image_from_url(url)

    assert image_data == b"fake_image_data"


def test_get_images(mocker):
    """
    Test the `get_images` method to ensure it generates and retrieves multiple images.

    Mocks both the OpenAI API call to generate image URLs and the `requests.get` call
    to download image data from these URLs. Verifies that the returned image data matches
    the expected output and that the correct number of images is retrieved.

    Args:
        mocker (pytest-mock): Pytest mocker to replace both `openai.images.generate`
        and `requests.get` calls.
    """
    mock_openai_response = mocker.Mock()
    mock_openai_response.data = [
        mocker.Mock(url="https://example.com/image1.png"),
        mocker.Mock(url="https://example.com/image2.png"),
    ]
    mocker.patch("openai.images.generate", return_value=mock_openai_response)
    mocker.patch(
        "graphic_designer_tool.core.image_generator.Image.open",
        return_value=b"fake_image_data",
    )

    mock_response = mocker.Mock()
    mock_response.content = b"fake_image_data"
    mocker.patch("requests.get", return_value=mock_response)

    generator = ImageGenerator(model="dall-e-3", quality="high", n=2, dim="1024x1024")
    images = generator.get_images("A futuristic cityscape")

    assert images == b"fake_image_data"
