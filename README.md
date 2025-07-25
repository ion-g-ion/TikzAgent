# TikZ Agent Workflow

A LangGraph-based agentic AI workflow for generating, compiling, and reviewing TikZ diagrams using large language models.

## Features

- **🤖 Multi-Agent System**: Intelligent workflow with specialized agents that collaborate: **Generate** → **Compile** → **Review**
- **🤖 Multi-LLM Support**: Works with OpenAI, Anthropic, Google, and other providers
- **📝 CLI Interface**: Easy-to-use command-line interface with script entry point
- **⚡ Async Support**: Both synchronous and asynchronous execution
- **🎨 Interactive Demo**: Streamlit-based web interface for easy experimentation
- **🔧 LaTeX Integration**: Built-in LaTeX compilation and PDF/PNG conversion tools

## Architecture

The workflow consists of three sequential nodes connected in a graph:

```
Generator → Compiler → Reviewer
    ↑                      ↓
    └─────── (conditional) ←┘
```

1. **Generator**: Creates TikZ code based on user requests or revision feedback
2. **Compiler**: Validates the generated code for syntax errors and issues using LaTeX compilation
3. **Reviewer**: Evaluates quality and determines if revisions are needed

The reviewer can either:
- ✅ **End** the workflow if the code meets quality standards
- 🔄 **Continue** back to the generator for revisions (up to max iterations)

## Installation

### Prerequisites

Make sure you have LaTeX installed on your system:
- **Linux**: `sudo apt-get install texlive-latex-extra texlive-tikz-extra`
- **macOS**: Install MacTeX from [https://www.tug.org/mactex/](https://www.tug.org/mactex/)
- **Windows**: Install MiKTeX from [https://miktex.org/](https://miktex.org/)

For PNG conversion (optional), install ImageMagick:
- **Linux**: `sudo apt-get install imagemagick`
- **macOS**: `brew install imagemagick`
- **Windows**: Download from [https://imagemagick.org/](https://imagemagick.org/)

### Install TikZ Agent

1. Clone the repository:
```bash
git clone <repository-url>
cd TikzAgent
```

2. Install the package:
```bash
# Basic installation
pip install -e .

# With demo dependencies
pip install -e .[demo]

# With all optional dependencies
pip install -e .[all]

# Development installation
pip install -e .[dev]
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

## Docker Installation

For a containerized setup with LaTeX and Python already configured:

### Quick Docker Setup

1. **Clone and build**:
```bash
git clone <repository-url>
cd TikzAgent
docker build -t tikz-agent .
```

2. **Run with environment variables**:
```bash
# Basic usage (runs on port 8501)
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your_openai_api_key" \
  tikz-agent

# Custom port and multiple providers
docker run -p 3000:3000 \
  -e PORT=3000 \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e ANTHROPIC_API_KEY="your_anthropic_api_key" \
  -e GOOGLE_API_KEY="your_google_api_key" \
  tikz-agent
```

3. **Or use Docker Compose** (recommended):
```bash
# Set your API keys in environment
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"  # optional
export GOOGLE_API_KEY="your_google_api_key"        # optional
export PORT=8501  # optional, defaults to 8501

# Run with compose
docker-compose up --build
```

4. **Or use convenience scripts**:
```bash
# Using the provided shell script
export OPENAI_API_KEY="your_key"
./docker-run.sh

# Using Make commands
make build          # Build the image
make run           # Run interactively  
make run-detached  # Run in background
make logs          # View logs
make stop          # Stop container
make help          # Show all commands
```

### Docker Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PORT` | Streamlit server port | `8501` | No |
| `OPENAI_API_KEY` | OpenAI API key | `""` | Optional* |
| `ANTHROPIC_API_KEY` | Anthropic API key | `""` | Optional* |
| `GOOGLE_API_KEY` | Google AI API key | `""` | Optional* |

*At least one API key is required to use the application.

**Tip**: Create a `.env` file in the project root with your API keys:
```bash
# .env file example
PORT=8501
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### Docker Features

- ✅ **Pre-installed LaTeX**: Based on `texlive/texlive:latest` with full TikZ support
- ✅ **Python 3.10**: Ready-to-use Python environment
- ✅ **ImageMagick**: For PNG conversion support
- ✅ **Security**: Runs as non-root user
- ✅ **Health Checks**: Built-in container health monitoring
- ✅ **Volume Support**: Mount `/app/output` for file persistence

### Advanced Docker Usage

```bash
# Run with output volume for file persistence
docker run -p 8501:8501 \
  -v $(pwd)/output:/app/output \
  -e OPENAI_API_KEY="your_key" \
  tikz-agent

# Run in detached mode with restart policy
docker run -d \
  --name tikz-agent \
  --restart unless-stopped \
  -p 8501:8501 \
  -e OPENAI_API_KEY="your_key" \
  tikz-agent

# Check logs
docker logs tikz-agent

# Stop container
docker stop tikz-agent
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

After installation, you can use the CLI through the installed script:

```bash
# Basic usage (using installed script)
tikz-agent "Create a binary tree with 7 nodes"

# Or run directly
python -m TikzAgent.cli "Create a binary tree with 7 nodes"

# Use a different provider
tikz-agent "Create a flowchart" --provider anthropic --model claude-3-sonnet-20240229

# Save output to file
tikz-agent "Create a neural network diagram" --output diagram.tikz

# Verbose output with custom settings
tikz-agent "Create a UML class diagram" --verbose --max-iterations 5 --temperature 0.5
```

### Programmatic Usage

```python
import asyncio
from langchain_openai import ChatOpenAI
from TikzAgent.workflow import create_tikz_workflow

async def main():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
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
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Create workflow
workflow = create_tikz_workflow(llm, max_iterations=3)

# Run workflow synchronously
result = workflow.run_sync("Create a TikZ diagram showing a simple flowchart")

print(f"Generated TikZ Code:\n{result['tikz_code']}")
```

### Streaming Usage

```python
from langchain_openai import ChatOpenAI
from TikzAgent.workflow import create_tikz_workflow

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Create workflow
workflow = create_tikz_workflow(llm, max_iterations=3)

# Stream through workflow steps
for chunk in workflow.stream_sync("Create a neural network diagram"):
    print(f"Node: {list(chunk.keys())[0]}")
    # Process each step as it completes
```

## Configuration

### Supported LLM Providers

| Provider | Models | Environment Variable | Optional Dependency |
|----------|---------|---------------------|-------------------|
| OpenAI | gpt-4, gpt-3.5-turbo, etc. | `OPENAI_API_KEY` | (included) |
| Anthropic | claude-3-sonnet-20240229, etc. | `ANTHROPIC_API_KEY` | (included) |
| Google | gemini-pro, etc. | `GOOGLE_API_KEY` | `pip install -e .[google]` |
| Mistral | mistral-large, etc. | `MISTRAL_API_KEY` | `pip install -e .[mistral]` |
| Groq | llama2-70b, etc. | `GROQ_API_KEY` | `pip install -e .[groq]` |

### Parameters

- **max_iterations**: Maximum number of revision cycles (default: 16)
- **temperature**: LLM temperature setting (default: 0.7)
- **model**: Specific model to use (varies by provider)

## LaTeX and Figure Tools

The package includes comprehensive LaTeX compilation and figure conversion tools:

```python
from TikzAgent.latex_tools import (
    compile_latex_to_pdf,
    Figure,
    save_figure_to_file,
    convert_pdf_figure_to_png,
    convert_png_figure_to_pdf,
    load_pdf_to_figure,
    check_latex_installation
)

# Check LaTeX installation
is_installed, version = check_latex_installation()
print(f"LaTeX installed: {is_installed}, Version: {version}")

# Compile LaTeX code
success, message, figure = compile_latex_to_pdf(latex_code)

# Convert between formats
png_figure = convert_pdf_figure_to_png(pdf_figure)
pdf_figure = convert_png_figure_to_pdf(png_figure)

# Save figures
save_figure_to_file(figure, "output.pdf")
```

## Examples

### Basic Geometric Shapes
```bash
tikz-agent "Create a pentagon with labeled vertices"
```

### Data Structures
```bash
tikz-agent "Create a binary search tree with nodes containing values 1, 3, 5, 7, 9"
```

### Flowcharts
```bash
tikz-agent "Create a flowchart showing a simple decision-making process with start, decision, process, and end nodes"
```

### Neural Networks
```bash
tikz-agent "Create a neural network diagram with 4 input nodes, 2 hidden layers with 6 and 4 nodes, and 2 output nodes"
```

### UML Diagrams
```bash
tikz-agent "Create a UML class diagram showing inheritance between Animal, Dog, and Cat classes"
```

## Error Handling

The workflow includes comprehensive error handling:

- **API Key Validation**: Checks for required environment variables
- **LaTeX Installation**: Verifies pdflatex availability
- **Compilation Errors**: Identifies TikZ syntax issues with detailed feedback
- **Iteration Limits**: Prevents infinite loops
- **LLM Errors**: Handles API failures gracefully

## Development

### Project Structure

```
TikzAgent/
├── TikzAgent/           # Main package
│   ├── __init__.py      # Package exports
│   ├── workflow.py      # Core LangGraph workflow
│   ├── latex_tools.py   # LaTeX compilation tools
│   └── cli.py          # Command-line interface
├── test/               # Test files
├── streamlit_demo.py   # Web demo interface
├── run_demo.py        # Demo launcher
├── pyproject.toml     # Package configuration
└── requirements.txt   # Dependencies
```

### Testing

```bash
# Install test dependencies
pip install -e .[test]

# Run tests
pytest test/

# Run with coverage
pytest --cov=TikzAgent test/
```

### Available Optional Dependencies

- `[demo]`: Streamlit demo dependencies
- `[google]`: Google AI support
- `[mistral]`: Mistral AI support  
- `[groq]`: Groq support
- `[all]`: All optional dependencies
- `[dev]`: Development tools (black, flake8, mypy, pre-commit)
- `[test]`: Testing dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `black` and `flake8` for code formatting
6. Submit a pull request

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

2. **LaTeX Not Installed**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install texlive-latex-extra texlive-tikz-extra
   
   # macOS
   brew install --cask mactex
   ```

3. **Import Errors**
   ```bash
   pip install -e .
   ```

4. **CLI Command Not Found**
   ```bash
   # Make sure package is installed
   pip install -e .
   
   # Or run directly
   python -m TikzAgent.cli "your request"
   ```

5. **Demo Dependencies Missing**
   ```bash
   pip install -e .[demo]
   ```

### Getting Help

- Check the CLI help: `tikz-agent --help`
- Test LaTeX installation: `python -c "from TikzAgent.latex_tools import check_latex_installation; print(check_latex_installation())"`
- Open an issue for bugs or feature requests
