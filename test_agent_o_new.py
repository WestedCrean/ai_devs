#!/usr/bin/env python3
"""
Test script for OAgent class
"""

import os
from src.ai_devs_core.agent import OAgent
from src.ai_devs_core.config import get_config


def test_agent_o_initialization():
    """Test that OAgent can be initialized with default model"""
    config = get_config()

    # Test default initialization
    agent = OAgent()
    assert agent.model_id == "gpt-3.5-turbo"
    print("✓ Default initialization works")

    # Test custom model initialization
    agent_custom = OAgent(model_id="gpt-4")
    assert agent_custom.model_id == "gpt-4"
    print("✓ Custom model initialization works")

    # Test custom endpoint initialization
    agent_endpoint = OAgent(
        model_id="gpt-3.5-turbo",
        api_base="https://custom.endpoint.ai/v1"
    )
    assert agent_endpoint.api_base == "https://custom.endpoint.ai/v1"
    print("✓ Custom endpoint initialization works")

    # Test that config is properly loaded
    assert agent.config.OPENAI_API_KEY is not None
    print("✓ Config loaded successfully")


def test_batch_job_not_implemented():
    """Test that batch_job raises NotImplementedError"""
    agent = OAgent()
    try:
        agent.batch_job()
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert str(e) == "Batch jobs are not implemented for OAgent"
        print("✓ batch_job raises NotImplementedError as expected")


def test_agent_o_methods_exist():
    """Test that OAgent has the expected methods"""
    agent = OAgent()

    # Check that required methods exist
    assert hasattr(agent, "chat_completion")
    assert hasattr(agent, "run_infer_on_each_row")
    assert hasattr(agent, "batch_job")
    assert hasattr(agent, "_generate_openai_tool")
    print("✓ All required methods exist")


def test_openai_tool_generation():
    """Test that OpenAI tool generation works correctly"""
    agent = OAgent()

    # Define a simple test function
    def test_function(name: str, age: int) -> str:
        """Test function for tool generation
        
        Args:
            name: The name parameter.
            age: The age parameter.
        """
        return f"Hello {name}, you are {age} years old"

    # Generate tool
    tool = agent._generate_openai_tool(test_function)

    # Verify tool structure
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "test_function"
    assert "Test function for tool generation" in tool["function"]["description"]
    assert "properties" in tool["function"]["parameters"]
    assert "name" in tool["function"]["parameters"]["properties"]
    assert "age" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["properties"]["name"]["type"] == "string"
    assert tool["function"]["parameters"]["properties"]["age"]["type"] == "integer"
    print("✓ OpenAI tool generation works correctly")


if __name__ == "__main__":
    print("Testing OAgent class...")
    test_agent_o_initialization()
    test_batch_job_not_implemented()
    test_agent_o_methods_exist()
    test_openai_tool_generation()
    print("\n✅ All tests passed!")