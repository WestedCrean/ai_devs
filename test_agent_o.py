#!/usr/bin/env python3
"""
Test script for ORAgent class
"""

import os
from src.ai_devs_core.agent import ORAgent
from src.ai_devs_core.config import get_config


def test_agent_o_initialization():
    """Test that ORAgent can be initialized with default model"""
    config = get_config()

    # Test default initialization
    agent = ORAgent()
    assert agent.model_id == "openai/gpt-5.2"
    print("✓ Default initialization works")

    # Test custom model initialization
    agent_custom = ORAgent(model_id="openai/gpt-4")
    assert agent_custom.model_id == "openai/gpt-4"
    print("✓ Custom model initialization works")

    # Test that config is properly loaded
    assert agent.config.OPENROUTER_API_KEY is not None
    print("✓ Config loaded successfully")


def test_batch_job_not_implemented():
    """Test that batch_job raises NotImplementedError"""
    agent = ORAgent()
    try:
        agent.batch_job()
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert str(e) == "Batch jobs are not implemented for ORAgent"
        print("✓ batch_job raises NotImplementedError as expected")


def test_agent_o_methods_exist():
    """Test that ORAgent has the expected methods"""
    agent = ORAgent()

    # Check that required methods exist
    assert hasattr(agent, "chat_completion")
    assert hasattr(agent, "run_infer_on_each_row")
    assert hasattr(agent, "batch_job")
    assert hasattr(agent, "_generate_openrouter_tool")
    print("✓ All required methods exist")


if __name__ == "__main__":
    print("Testing ORAgent class...")
    test_agent_o_initialization()
    test_batch_job_not_implemented()
    test_agent_o_methods_exist()
    print("\n✅ All tests passed!")
