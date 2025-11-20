#!/usr/bin/env python3
"""
Test runner script that explicitly runs only llm_agents tests, excluding gradio.
"""
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run pytest with explicit test path to avoid gradio conflicts."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",  # Explicit path
        "--cov=llm_agents",
        "--cov-report=term-missing", 
        "--cov-report=html:htmlcov",
        "--cov-config=.coveragerc",
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())