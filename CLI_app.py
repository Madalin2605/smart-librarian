"""
Smart Librarian - Command Line Interface (CLI) entrypoint.

This script imports and runs the `run_cli()` function from `chatbot.interface`,
which provides an interactive text-based conversation loop in the terminal.
Use this file when you want to interact with Smart Librarian via CLI
instead of the Streamlit web UI.
"""

from chatbot.interface import run_cli


if __name__ == "__main__":

    run_cli()
