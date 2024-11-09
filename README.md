# Graphic Designer Tool

This is a small project leveraging a OpenAI Dalle to generate logo designs

## Requirements
* Python 3.10
* OpenAI API token

## Installation
 To ensure running the project smoothly, I recommend to set up a virtual environment. You can do so by running
`make setup-venv`. This will create your virtual environment. You can then run `source venv/bin/activate` to activate
 it (Not mandatory).

The project dependencies can be installed running `make install` in your terminal

## Open API Key
An Open API key is mandatory to be able to use the LLM model. Follow these steps https://platform.openai.com/docs/quickstart.


You can check that the OpenAI API key was successfully imported configured with `echo $OPENAI_API_KEY`

## Run
A demo notebook is located in demo/.

You can spin up the UI with `make run`





