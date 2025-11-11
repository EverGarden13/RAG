"""
User interface module for RAG system.
Includes terminal and web-based interfaces.
"""

from src.ui.terminal_interface import TerminalInterface, create_terminal_interface
from src.ui.web_interface import WebInterface, create_web_interface, run_streamlit_app
from src.ui.flask_interface import FlaskInterface, create_flask_interface

__all__ = [
    'TerminalInterface',
    'create_terminal_interface',
    'WebInterface', 
    'create_web_interface',
    'run_streamlit_app',
    'FlaskInterface',
    'create_flask_interface'
]
