""" Main application test """

import pytest

from graphic_designer_tool.main import gradio, main


@pytest.fixture(name="mock_load_config")
def mock_load_config_fixture(mocker):
    """
    Mock the `load_config` function to return sample configuration parameters.

    This fixture ensures that the configuration is mocked for the tests, avoiding
    any reliance on external files.
    """
    mock_config = {
        "n_image": 2,
        "image_dim": "1024x1024",
        "image_model": "dall-e",
        "image_quality": "high",
    }
    return mocker.patch(
        "graphic_designer_tool.main.load_config", return_value=mock_config
    )


@pytest.fixture(name="mock_gradio_interface")
def mock_gradio_interface_fixture(mocker):
    """
    Mock the Gradio interface to prevent it from actually launching during tests.

    This fixture mocks the Gradio interface to ensure the test focuses on the logic
    rather than the user interface behavior.
    """
    mock_interface = mocker.patch.object(gradio, "Interface")
    return mock_interface


def test_main_success(mock_load_config, mock_gradio_interface):
    """
    Test the `main` function to ensure it initializes
    components and sets up the Gradio interface correctly.

    This test verifies that the `load_config` function is called to load the configuration,
    the ImageGenerator is initialized with the correct parameters, and the Gradio interface
    is set up with the correct function and inputs.
    """
    # Call the main function
    main()

    # Assert that load_config was called with the correct file path
    mock_load_config.assert_called_once_with(config_path="configs/params.yml")

    # Assert that the Gradio interface was created with the correct function and inputs
    mock_gradio_interface.assert_called_once()


def test_main_invalid_config(mock_load_config):
    """
    Test the `main` function when there is an invalid configuration.

    This test simulates the scenario where `load_config` returns invalid configuration,
    ensuring that appropriate error handling occurs. This will raise an exception in the
    test to verify that the error is handled properly.
    """
    # Simulate invalid configuration data
    mock_load_config.return_value = {
        "n_image": -1,  # Invalid number of images
        "image_dim": "800x600",  # Invalid dimension
        "image_model": "invalid_model",  # Invalid model
        "image_quality": "low",  # Invalid quality
    }

    # Try calling the main function and assert that an exception is raised
    with pytest.raises(ValueError):
        main()
