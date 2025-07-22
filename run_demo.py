#!/usr/bin/env python3
"""
TikZ Agent Demo Launcher

Simple script to run the Streamlit demo with proper configuration.
"""

import subprocess
import sys
import os


def main():
    """Launch the Streamlit demo."""
    print("🚀 Starting TikZ Agent Demo...")
    print("📱 Opening web interface at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # Run streamlit with the demo file
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_demo.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install dependencies first:")
        print("   pip install streamlit")
        print("   or")
        print("   pip install -e .[demo]")
        sys.exit(1)


if __name__ == "__main__":
    main() 