""" Test config core functions """
import pytest

from graphic_designer_tool.core.config import (
    load_config,
)  # Replace with the actual module name


@pytest.fixture(name="valid_yaml_file")
def valid_yaml_file_fixture(tmp_path):
    """Fixture to create a valid YAML file for testing."""
    yaml_content = """
    text_splitter_chunk_size: 100
    text_splitter_chunk_overlap: 10
    text_splitter_separators: [",", ";"]
    document_path: "path/to/document.pdf"
    llm_model_name: "gpt-3"
    embedding_path: "path/to/embedding"
    """
    yaml_file = tmp_path / "params.yml"
    yaml_file.write_text(yaml_content, encoding="utf-8")
    return yaml_file


def test_load_config_valid(valid_yaml_file):
    """Test loading a valid YAML configuration file."""
    config = load_config(str(valid_yaml_file))
    assert isinstance(config, dict)
    assert config["text_splitter_chunk_size"] == 100
    assert config["document_path"] == "path/to/document.pdf"


def test_load_config_file_not_found():
    """Test loading a non-existent YAML configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yml")
