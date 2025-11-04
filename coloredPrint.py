"""
Colored print utility for better terminal output visualization.
Supports ANSI color codes (Unix/Mac) and colorama for Windows compatibility.
"""

import sys

# Try to import colorama for Windows support, fallback to ANSI codes
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)  # Automatically reset after each print
    USE_COLORAMA = True
except ImportError:
    USE_COLORAMA = False
    # ANSI color codes
    class Fore:
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        RESET = '\033[0m'
    
    class Back:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        MAGENTA = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        RESET = '\033[0m'
    
    class Style:
        BRIGHT = '\033[1m'
        DIM = '\033[2m'
        NORMAL = '\033[22m'
        RESET_ALL = '\033[0m'


def colored_print(message, color=Fore.WHITE, style=Style.NORMAL, end='\n', file=sys.stdout):
    """
    Print a colored message.
    
    Args:
        message: Message to print
        color: Color from Fore (e.g., Fore.RED, Fore.GREEN)
        style: Style from Style (e.g., Style.BRIGHT)
        end: End character (default: newline)
        file: Output file (default: stdout)
    """
    if not USE_COLORAMA:
        # For ANSI codes, need to reset at the end
        print(f"{style}{color}{message}{Style.RESET_ALL}", end=end, file=file)
    else:
        print(f"{style}{color}{message}", end=end, file=file)


# Convenience functions for common use cases
def print_success(message, **kwargs):
    """Print a success message in green."""
    colored_print(message, color=Fore.GREEN, style=Style.BRIGHT, **kwargs)


def print_error(message, **kwargs):
    """Print an error message in red."""
    colored_print(message, color=Fore.RED, style=Style.BRIGHT, **kwargs)


def print_warning(message, **kwargs):
    """Print a warning message in yellow."""
    colored_print(message, color=Fore.YELLOW, style=Style.BRIGHT, **kwargs)


def print_info(message, **kwargs):
    """Print an info message in cyan."""
    colored_print(message, color=Fore.CYAN, **kwargs)


def print_debug(message, **kwargs):
    """Print a debug message in blue."""
    colored_print(message, color=Fore.BLUE, **kwargs)


def print_highlight(message, **kwargs):
    """Print a highlighted message in magenta."""
    colored_print(message, color=Fore.MAGENTA, style=Style.BRIGHT, **kwargs)


def print_session(message, **kwargs):
    """Print a session-related message in bright cyan."""
    colored_print(message, color=Fore.CYAN, style=Style.BRIGHT, **kwargs)


def print_trial(message, **kwargs):
    """Print a trial-related message in bright green."""
    colored_print(message, color=Fore.GREEN, style=Style.NORMAL, **kwargs)


def print_data(message, **kwargs):
    """Print data-related message in blue."""
    colored_print(message, color=Fore.BLUE, style=Style.NORMAL, **kwargs)



# Example usage
if __name__ == "__main__":
    print_success("✓ Success: This is a success message")
    print_error("✗ Error: This is an error message")
    print_warning("⚠ Warning: This is a warning message")
    print_info("ℹ Info: This is an info message")
    print_debug("🔍 Debug: This is a debug message")
    print_highlight("🌟 Highlight: This is a highlighted message")
    print_session("📊 Session: This is a session message")
    print_trial("🎯 Trial: This is a trial message")
    print_data("💾 Data: This is a data message")
    
    # You can also use colored_print directly for custom colors
    colored_print("Custom: This is a custom colored message", color=Fore.MAGENTA, style=Style.BRIGHT)

