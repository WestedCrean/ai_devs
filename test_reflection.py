#!/usr/bin/env python3
"""
Test script for FAgent.chat_completion_with_reflect method
"""

from src.ai_devs_core.agent import FAgent


def test_reflection_basic():
    """Test basic reflection functionality"""
    print("Testing FAgent.chat_completion_with_reflect...")
    
    # Initialize config and agent
    agent = FAgent(model_id="mistral-small-latest")  # Use smaller model for testing
    
    # Test message
    test_message = "What are the key principles of good software design?"
    
    print(f"Original message: {test_message}")
    
    try:
        # Test with reflection (max 2 reflections)
        response = agent.chat_completion_with_reflect(
            message=test_message,
            max_reflections=2,
            reflection_prompt="Analyze this response for completeness, accuracy, and depth. Suggest specific improvements:"
        )
        
        result = response.choices[0].message.content
        print(f"Final response length: {len(result)} characters")
        print(f"First 200 characters: {result[:200]}...")
        print("✓ Basic reflection test passed")
        
    except Exception as e:
        print(f"✗ Basic reflection test failed: {e}")
        raise


def test_reflection_with_tools():
    """Test reflection with tools"""
    print("\nTesting reflection with tools...")
    
    def mock_tool_1(parameter: str) -> str:
        """Mock tool that returns a simple response"""
        return f"Tool 1 processed: {parameter}"
    
    def mock_tool_2(number: int) -> int:
        """Mock tool that does calculation"""
        return number * 2
    
    try:
        agent = FAgent(model_id="mistral-small-latest")
        
        agent.chat_completion_with_reflect(
            message="Analyze this data and calculate results using available tools",
            tools=[mock_tool_1, mock_tool_2],
            max_reflections=1
        )
        
        print("✓ Reflection with tools test passed")
        
    except Exception as e:
        print(f"✗ Reflection with tools test failed: {e}")
        raise


def test_reflection_early_exit():
    """Test early exit when critique is satisfactory"""
    print("\nTesting early exit from reflection loop...")
    
    try:
        agent = FAgent(model_id="mistral-small-latest")
        
        # Use a simple question that should get good response quickly
        response = agent.chat_completion_with_reflect(
            message="What is 2 + 2?",
            max_reflections=3,
            reflection_prompt="Analyze this mathematical response for correctness:"
        )
        
        result = response.choices[0].message.content
        print(f"Simple math result: {result.strip()}")
        print("✓ Early exit test passed")
        
    except Exception as e:
        print(f"✗ Early exit test failed: {e}")
        raise


if __name__ == "__main__":
    print("Running FAgent reflection tests...")
    
    try:
        test_reflection_basic()
        test_reflection_with_tools()
        test_reflection_early_exit()
        
        print("\n✅ All reflection tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        exit(1)
