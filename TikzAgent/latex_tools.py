import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Union, Literal
from pydantic import BaseModel

class Figure(BaseModel):
    """Typed figure for the workflow."""
    type: Literal["png", "pdf"]
    data: bytes
    latex_code: str
    
def compile_latex_to_pdf(latex_code: str) -> Tuple[bool, str, Optional[Figure]]:
    """
    Compile LaTeX code to PDF using pdflatex.
    
    Args:
        latex_code (str): The LaTeX code as a string
    
    Returns:
        Tuple[bool, str, Optional[Figure]]: 
            - bool: True if compilation successful, False otherwise
            - str: Status message or error message
            - Optional[Figure]: Figure object with PDF data if successful, None if failed
    """
    # Create temporary directory for compilation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tex_file = temp_path / "document.tex"
        pdf_file = temp_path / "document.pdf"
        
        try:
            # Write LaTeX code to temporary file
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            
            # Run pdflatex
            result = subprocess.run([
                'pdflatex', 
                '-interaction=nonstopmode',  # Don't stop on errors
                '-output-directory', str(temp_path),
                str(tex_file)
            ], 
            capture_output=True, 
            text=True, 
            cwd=temp_path
            )
            
            # Check if PDF was generated successfully
            if pdf_file.exists() and result.returncode == 0:
                # Read the PDF file
                with open(pdf_file, 'rb') as f:
                    pdf_content = f.read()
                
                # Create Figure object
                figure = Figure(
                    type="pdf",
                    data=pdf_content,
                    latex_code=latex_code
                )
                
                return True, "LaTeX compilation successful", figure
                
            else:
                # Compilation failed
                error_msg = f"pdflatex failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                return False, error_msg, None
                
        except FileNotFoundError:
            return False, "pdflatex not found. Please install a LaTeX distribution (e.g., TeX Live, MiKTeX)", None
        except Exception as e:
            return False, f"Unexpected error during compilation: {str(e)}", None

def save_figure_to_file(figure: Figure, filepath: Union[str, Path]) -> bool:
    """
    Save Figure content to a file.
    
    Args:
        figure (Figure): Figure object containing the data to save
        filepath (Union[str, Path]): Path where to save the file
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add appropriate extension if not present
        if not filepath.suffix:
            filepath = filepath.with_suffix(f".{figure.type}")
        
        with open(filepath, 'wb') as f:
            f.write(figure.data)
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False


def check_latex_installation() -> Tuple[bool, str]:
    """
    Check if pdflatex is installed and accessible.
    
    Returns:
        Tuple[bool, str]: (is_installed, version_info)
    """
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.split('\n')[0]
        else:
            return False, "pdflatex found but returned error"
    except FileNotFoundError:
        return False, "pdflatex not found"
    except Exception as e:
        return False, f"Error checking installation: {e}"

def convert_pdf_figure_to_png(pdf_figure: Figure, dpi: int = 300) -> Figure:
    """
    Convert a PDF figure to PNG format using ImageMagick.
    
    Args:
        pdf_figure (Figure): Figure object containing PDF data
        dpi (int): Resolution for PNG conversion (default: 300)
    
    Returns:
        Figure: Figure object with PNG data
        
    Raises:
        ValueError: If input figure is not of type 'pdf'
        FileNotFoundError: If ImageMagick is not installed
        RuntimeError: If conversion fails
    """
    if pdf_figure.type != 'pdf':
        raise ValueError("Input figure must be of type 'pdf'")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_file = temp_path / "input.pdf"
        png_file = temp_path / "output.png"
        
        try:
            # Write PDF data to temporary file
            with open(pdf_file, 'wb') as f:
                f.write(pdf_figure.data)
            
            # Convert PDF to PNG using ImageMagick
            result = subprocess.run([
                'convert',
                '-density', str(dpi),
                '-background', 'white',
                '-alpha', 'remove',
                '-quality', '100',
                '-colors', '255',
                str(pdf_file),
                str(png_file)
            ], 
            capture_output=True, 
            text=True
            )
            
            # Check if PNG was generated successfully
            if png_file.exists() and result.returncode == 0:
                # Read the PNG file
                with open(png_file, 'rb') as f:
                    png_content = f.read()
                
                # Check if PNG file is not empty
                if len(png_content) == 0:
                    error_msg = f"ImageMagick generated empty PNG file\n"
                    error_msg += f"STDOUT: {result.stdout}\n"
                    error_msg += f"STDERR: {result.stderr}\n"
                    error_msg += f"PDF file size: {len(pdf_figure.data)} bytes"
                    raise RuntimeError(error_msg)
                
                # Create new Figure object with PNG data
                png_figure = Figure(
                    type="png",
                    data=png_content,
                    latex_code=pdf_figure.latex_code
                )
                
                return png_figure
                
            else:
                # Conversion failed
                error_msg = f"ImageMagick convert failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                raise RuntimeError(error_msg)
                
        except FileNotFoundError:
            raise FileNotFoundError("ImageMagick 'convert' not found. Please install ImageMagick")
        except (ValueError, FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during conversion: {str(e)}")


def load_pdf_to_figure(filepath: Union[str, Path], latex_code: str = "") -> Figure:
    """
    Load a PDF file from disk and create a Figure object.
    
    Args:
        filepath (Union[str, Path]): Path to the PDF file to load
        latex_code (str): Optional LaTeX code that generated this PDF (default: empty string)
    
    Returns:
        Figure: Figure object with PDF data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file is not a PDF
        RuntimeError: If there's an error reading the PDF file
    """
    filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check if it's a PDF file
    if filepath.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {filepath}")
    
    try:
        # Read the PDF file
        with open(filepath, 'rb') as f:
            pdf_content = f.read()
        
        # Create Figure object
        figure = Figure(
            type="pdf",
            data=pdf_content,
            latex_code=latex_code
        )
        
        return figure
        
    except Exception as e:
        raise RuntimeError(f"Error loading PDF file: {str(e)}")


def convert_png_figure_to_pdf(png_figure: Figure) -> Figure:
    """
    Convert a PNG figure to PDF format using ImageMagick.
    
    Args:
        png_figure (Figure): Figure object containing PNG data
    
    Returns:
        Figure: Figure object with PDF data
        
    Raises:
        ValueError: If input figure is not of type 'png'
        FileNotFoundError: If ImageMagick is not installed
        RuntimeError: If conversion fails
    """
    if png_figure.type != 'png':
        raise ValueError("Input figure must be of type 'png'")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        png_file = temp_path / "input.png"
        pdf_file = temp_path / "output.pdf"
        
        try:
            # Write PNG data to temporary file
            with open(png_file, 'wb') as f:
                f.write(png_figure.data)
            
            # Convert PNG to PDF using ImageMagick
            result = subprocess.run([
                'convert',
                str(png_file),
                str(pdf_file)
            ], 
            capture_output=True, 
            text=True
            )
            
            # Check if PDF was generated successfully
            if pdf_file.exists() and result.returncode == 0:
                # Read the PDF file
                with open(pdf_file, 'rb') as f:
                    pdf_content = f.read()
                
                # Create new Figure object with PDF data
                pdf_figure = Figure(
                    type="pdf",
                    data=pdf_content,
                    latex_code=png_figure.latex_code
                )
                
                return pdf_figure
                
            else:
                # Conversion failed
                error_msg = f"ImageMagick convert failed with return code {result.returncode}\n"
                error_msg += f"STDOUT: {result.stdout}\n"
                error_msg += f"STDERR: {result.stderr}"
                raise RuntimeError(error_msg)
                
        except FileNotFoundError:
            raise FileNotFoundError("ImageMagick 'convert' not found. Please install ImageMagick")
        except (ValueError, FileNotFoundError, RuntimeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during conversion: {str(e)}")

