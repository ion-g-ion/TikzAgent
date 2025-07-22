"""
Example usage of the TikZ Agent Workflow with different LLM providers.
"""

import asyncio
import os
from workflow import create_tikz_workflow


async def example_with_openai():
    """Example using OpenAI GPT models."""
    from langchain_openai import ChatOpenAI
    
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")  # Set your API key as environment variable
    )
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=3)
    
    # Example request
    user_request = "Create a TikZ diagram showing a binary tree with 7 nodes"
    
    # Run workflow
    result = await workflow.run(user_request)
    
    print("\n" + "="*50)
    print("OPENAI RESULT:")
    print("="*50)
    print(f"Final TikZ Code:\n{result['tikz_code']}")
    print(f"\nIterations: {result['iteration_count']}")


async def example_with_anthropic():
    """Example using Anthropic Claude models."""
    from langchain_anthropic import ChatAnthropic
    
    # Initialize Anthropic LLM
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")  # Set your API key as environment variable
    )
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=3)
    
    # Example request
    user_request = "Create a TikZ diagram showing a flowchart for a simple decision-making process"
    
    # Run workflow
    result = await workflow.run(user_request)
    
    print("\n" + "="*50)
    print("ANTHROPIC RESULT:")
    print("="*50)
    print(f"Final TikZ Code:\n{result['tikz_code']}")
    print(f"\nIterations: {result['iteration_count']}")


async def example_with_google():
    """Example using Google Gemini models."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Initialize Google LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # Set your API key as environment variable
    )
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=3)
    
    # Example request
    user_request = "Create a TikZ diagram showing a UML class diagram with 3 classes and their relationships"
    
    # Run workflow
    result = await workflow.run(user_request)
    
    print("\n" + "="*50)
    print("GOOGLE RESULT:")
    print("="*50)
    print(f"Final TikZ Code:\n{result['tikz_code']}")
    print(f"\nIterations: {result['iteration_count']}")


def synchronous_example():
    """Example of using the synchronous version of the workflow."""
    from langchain_openai import ChatOpenAI
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=2)
    
    # Example request
    user_request = "Create a TikZ diagram showing a simple geometric shape like a pentagon"
    
    # Run workflow synchronously
    result = workflow.run_sync(user_request)
    
    print("\n" + "="*50)
    print("SYNCHRONOUS RESULT:")
    print("="*50)
    print(f"Final TikZ Code:\n{result['tikz_code']}")
    print(f"\nIterations: {result['iteration_count']}")


async def custom_example():
    """Example with custom parameters and detailed output."""
    from langchain_openai import ChatOpenAI
    
    # Initialize LLM with custom settings
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.5,  # Lower temperature for more consistent output
        max_tokens=2000,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create workflow with custom max iterations
    workflow = create_tikz_workflow(llm, max_iterations=5)
    
    # Complex example request
    user_request = """
    Create a TikZ diagram showing a complete neural network architecture with:
    - Input layer with 4 nodes
    - Two hidden layers with 6 and 4 nodes respectively
    - Output layer with 2 nodes
    - All connections between layers
    - Labels for each layer
    - Different colors for different layers
    """
    
    # Run workflow
    result = await workflow.run(user_request)
    
    print("\n" + "="*50)
    print("CUSTOM EXAMPLE RESULT:")
    print("="*50)
    print(f"Final TikZ Code:\n{result['tikz_code']}")
    print(f"\nCompilation Result:\n{result['compilation_result']}")
    print(f"\nReview Result:\n{result['review_result']}")
    print(f"\nTotal Iterations: {result['iteration_count']}")
    print(f"\nTotal Messages: {len(result['messages'])}")


async def main():
    """Main function to run examples."""
    print("🚀 TikZ Agent Workflow Examples")
    print("="*50)
    
    # Check for API keys
    available_providers = []
    if os.getenv("OPENAI_API_KEY"):
        available_providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append("Anthropic")
    if os.getenv("GOOGLE_API_KEY"):
        available_providers.append("Google")
    
    print(f"Available providers: {', '.join(available_providers) if available_providers else 'None (set API keys)'}")
    
    # Run examples based on available providers
    if "OpenAI" in available_providers:
        await example_with_openai()
        await custom_example()
    
    if "Anthropic" in available_providers:
        await example_with_anthropic()
    
    if "Google" in available_providers:
        await example_with_google()
    
    # Always run synchronous example if OpenAI is available
    if "OpenAI" in available_providers:
        synchronous_example()
    
    if not available_providers:
        print("\n⚠️  No API keys found. Please set at least one of:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY")
        print("- GOOGLE_API_KEY")


if __name__ == "__main__":
    asyncio.run(main()) 