# TikZ Agent Workflow

A LangGraph-based agentic AI workflow for generating, compiling, and reviewing TikZ diagrams using large language models.

## Features

- **🎨 Generator Node**: Generates TikZ code based on user requests
- **🔧 Compiler Node**: Validates and analyzes TikZ code for errors
- **👁️ Reviewer Node**: Reviews output quality and decides on revisions
- **🔄 Iterative Refinement**: Automatically improves code through multiple iterations
- **🤖 Multi-LLM Support**: Works with OpenAI, Anthropic, Google, and other providers
- **📝 CLI Interface**: Easy-to-use command-line interface
- **⚡ Async Support**: Both synchronous and asynchronous execution

## Architecture

The workflow consists of three sequential nodes connected in a graph:

```
Generator → Compiler → Reviewer
    ↑                      ↓
    └─────── (conditional) ←┘
```

1. **Generator**: Creates TikZ code based on user requests or revision feedback
2. **Compiler**: Validates the generated code for syntax errors and issues
3. **Reviewer**: Evaluates quality and determines if revisions are needed

The reviewer can either:
- ✅ **End** the workflow if the code meets quality standards
- 🔄 **Continue** back to the generator for revisions (up to max iterations)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TikzAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
# For OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# For Anthropic
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# For Google
export GOOGLE_API_KEY="your_google_api_key"
```

## Quick Start - Streamlit Demo

The fastest way to get started is with the interactive Streamlit demo:

```bash
# Install with demo dependencies
pip install -e .[demo]

# Run the demo
python run_demo.py
```

Or run directly:
```bash
streamlit run streamlit_demo.py
```

The demo provides:
- 🎨 **Interactive Web Interface**: User-friendly form for TikZ generation
- 🔧 **Multiple LLM Providers**: Choose between OpenAI, Anthropic, or Google
- 📝 **Example Templates**: Pre-built examples to get started quickly
- 📊 **Real-time Progress**: See generation, compilation, and review steps
- 📄 **PDF Download**: Download generated diagrams as PDF files
- 🎯 **Visual Results**: View generated TikZ code and compilation results

## Usage

### Command Line Interface

The easiest way to use the TikZ Agent is through the CLI:

```bash
# Basic usage
python TikzAgent/cli.py "Create a binary tree with 7 nodes"

# Use a different provider
python TikzAgent/cli.py "Create a flowchart" --provider anthropic --model claude-3-sonnet-20240229

# Save output to file
python TikzAgent/cli.py "Create a neural network diagram" --output diagram.tikz

# Verbose output with custom settings
python TikzAgent/cli.py "Create a UML class diagram" --verbose --max-iterations 5 --temperature 0.5
```

### Programmatic Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from TikzAgent.workflow import create_tikz_workflow

async def main():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=3)
    
    # Run workflow
    result = await workflow.run("Create a TikZ diagram showing a binary tree")
    
    print(f"Generated TikZ Code:\n{result['tikz_code']}")
    print(f"Iterations: {result['iteration_count']}")

asyncio.run(main())
```

### Synchronous Usage

```python
from langchain_openai import ChatOpenAI
from TikzAgent.workflow import create_tikz_workflow

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create workflow
workflow = create_tikz_workflow(llm, max_iterations=3)

# Run workflow synchronously
result = workflow.run_sync("Create a TikZ diagram showing a simple flowchart")

print(f"Generated TikZ Code:\n{result['tikz_code']}")
```

## Configuration

### Supported LLM Providers

| Provider | Models | Environment Variable |
|----------|---------|---------------------|
| OpenAI | gpt-4, gpt-3.5-turbo, etc. | `OPENAI_API_KEY` |
| Anthropic | claude-3-sonnet-20240229, etc. | `ANTHROPIC_API_KEY` |
| Google | gemini-pro, etc. | `GOOGLE_API_KEY` |

### Parameters

- **max_iterations**: Maximum number of revision cycles (default: 3)
- **temperature**: LLM temperature setting (default: 0.7)
- **model**: Specific model to use (varies by provider)

## Examples

### Basic Geometric Shapes
```bash
python TikzAgent/cli.py "Create a pentagon with labeled vertices"
```

### Data Structures
```bash
python TikzAgent/cli.py "Create a binary search tree with nodes containing values 1, 3, 5, 7, 9"
```

### Flowcharts
```bash
python TikzAgent/cli.py "Create a flowchart showing a simple decision-making process with start, decision, process, and end nodes"
```

### Neural Networks
```bash
python TikzAgent/cli.py "Create a neural network diagram with 4 input nodes, 2 hidden layers with 6 and 4 nodes, and 2 output nodes"
```

### UML Diagrams
```bash
python TikzAgent/cli.py "Create a UML class diagram showing inheritance between Animal, Dog, and Cat classes"
```

## Workflow State

The workflow maintains state throughout execution:

```python
{
    "messages": [],           # Conversation history
    "tikz_code": "",         # Generated TikZ code
    "compilation_result": "", # Compiler analysis
    "review_result": "",     # Reviewer feedback
    "needs_revision": False, # Whether revision is needed
    "iteration_count": 0,    # Current iteration
    "max_iterations": 3      # Maximum iterations
}
```

## Error Handling

The workflow includes comprehensive error handling:

- **API Key Validation**: Checks for required environment variables
- **Compilation Errors**: Identifies TikZ syntax issues
- **Iteration Limits**: Prevents infinite loops
- **LLM Errors**: Handles API failures gracefully

## Development

### Running Examples

```bash
# Run all examples
python TikzAgent/example_usage.py

# Run specific provider examples
OPENAI_API_KEY=your_key python TikzAgent/example_usage.py
```

### Testing

```bash
pip install pytest pytest-asyncio
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- Inspired by the need for automated TikZ diagram generation

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

2. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

3. **Async Issues**
   - Use `asyncio.run()` for async functions
   - Use `run_sync()` method for synchronous execution

### Getting Help

- Check the [examples](TikzAgent/example_usage.py) for usage patterns
- Review the [CLI help](TikzAgent/cli.py) for available options
- Open an issue for bugs or feature requests
