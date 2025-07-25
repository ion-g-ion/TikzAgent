"""
Command-line interface for the TikZ Agent Workflow.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

try:
    from TikzAgent.workflow import create_tikz_workflow
except ImportError:
    from .workflow import create_tikz_workflow


def get_llm_from_provider(provider: str, model: str, temperature: float = 0.7):
    """
    Get an LLM instance based on the provider.
    
    Args:
        provider: The LLM provider ('openai', 'anthropic', 'google')
        model: The model name
        temperature: The temperature setting
        
    Returns:
        Configured LLM instance
    """
    if provider.lower() == 'openai':
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    
    elif provider.lower() == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)
    
    elif provider.lower() == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def run_workflow(
    request: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_iterations: int = 3,
    output_file: Optional[str] = None,
    verbose: bool = False
):
    """
    Run the TikZ workflow with the given parameters.
    
    Args:
        request: The user's request for TikZ code generation
        provider: The LLM provider to use
        model: The model name
        temperature: The temperature setting
        max_iterations: Maximum number of iterations
        output_file: Optional file to save the TikZ code
        verbose: Whether to show verbose output
    """
    try:
        # Get the LLM
        llm = get_llm_from_provider(provider, model, temperature)
        
        # Create and run the workflow
        workflow = create_tikz_workflow(llm, max_iterations)
        
        if verbose:
            print(f"🚀 Starting TikZ workflow with {provider} {model}")
            print(f"📝 Request: {request}")
            print(f"🔧 Max iterations: {max_iterations}")
            print("-" * 50)
        
        # Run the workflow
        result = await workflow.run(request)
        
        # Display results
        print("\n" + "="*50)
        print("✅ WORKFLOW COMPLETED")
        print("="*50)
        
        if verbose:
            print(f"🔄 Iterations: {result['iteration_count']}")
            print(f"📊 Compilation Result:\n{result['compilation_result']}")
            print(f"👁️ Review Result:\n{result['review_result']}")
            print("-" * 50)
        
        print("🎨 Generated TikZ Code:")
        print("-" * 30)
        print(result['tikz_code'])
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result['tikz_code'])
            print(f"\n💾 TikZ code saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="TikZ Agent Workflow - Generate TikZ code using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenAI
  python cli.py "Create a binary tree with 7 nodes"
  
  # Use Anthropic Claude
  python cli.py "Create a flowchart" --provider anthropic --model claude-3-sonnet-20240229
  
  # Save output to file
  python cli.py "Create a neural network diagram" --output diagram.tikz
  
  # Verbose output with custom settings
  python cli.py "Create a UML class diagram" --verbose --max-iterations 5 --temperature 0.5
        """
    )
    
    parser.add_argument(
        "request",
        help="The TikZ generation request"
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature setting (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of iterations (default: 3)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file to save the TikZ code"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    required_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY"
    }
    
    env_var = required_env_vars.get(args.provider)
    if env_var and not os.getenv(env_var):
        print(f"❌ Error: {env_var} environment variable not set")
        print(f"Please set your API key: export {env_var}=your_api_key_here")
        sys.exit(1)
    
    # Run the workflow
    result = asyncio.run(run_workflow(
        request=args.request,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        output_file=args.output,
        verbose=args.verbose
    ))
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main() 