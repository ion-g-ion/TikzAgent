"""TikzAgent: A LangGraph-based agentic AI workflow for generating TikZ diagrams."""

__version__ = "0.1.0"
__author__ = "Ion Gabriel Ion"
__email__ = "ion.ion.gabriel@gmail.com"
__description__ = "A LangGraph-based agentic AI workflow for generating, compiling, and reviewing TikZ diagrams using large language models"

# Main exports
from .workflow import create_tikz_workflow
from .latex_tools import compile_latex_to_pdf, Figure, save_figure_to_file, check_latex_installation, convert_pdf_figure_to_png, load_pdf_to_figure, convert_png_figure_to_pdf

__all__ = [
    "create_tikz_workflow",
    "compile_latex_to_pdf",
    "Figure",
    "save_figure_to_file",
    "check_latex_installation",
    "convert_pdf_figure_to_png",
    "load_pdf_to_figure",
    "convert_png_figure_to_pdf",
]
    
    
    
    
    