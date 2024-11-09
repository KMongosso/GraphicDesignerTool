""" Main application """

import gradio

from graphic_designer_tool.core.config import load_config
from graphic_designer_tool.core.image_generator import ImageGenerator


def main() -> None:
    """
    Main function to launch the Graphic Designer Tool application.

    This function:
    1. Loads configuration parameters from a YAML file.
    2. Initializes the Image Generator with the specified parameters.
    3. Sets up and launches a Gradio interface to allow the user to generate visuals
       by providing a textual description.

    The Gradio interface facilitates the generation of images by interacting with
    the Image Generator, which uses a pre-configured model to create visuals based
    on user input. The user is prompted to provide a description, and the tool
    returns a generated image.

    Workflow:
    1. Load configuration settings (number of images, image dimensions, model, quality).
    2. Initialize the Image Generator with these parameters.
    3. Launch a Gradio interface that accepts user input and displays generated images.

    Returns:
        None
    """
    params = load_config(config_path="configs/params.yml")

    # Initialize Image Generator
    image_generator = ImageGenerator(
        n=params["n_image"],
        dim=params["image_dim"],
        model=params["image_model"],
        quality=params["image_quality"],
    )

    # Set up the Gradio interface
    demo = gradio.Interface(
        fn=image_generator.get_images,
        inputs=gradio.Textbox(
            label="Visual description",
            placeholder="Type the description of the visual you want to create here here...",
        ),
        outputs=gradio.Image(label="Generated visual"),
        title="Graphic designer tool",
    )
    demo.launch()


if __name__ == "__main__":
    main()
