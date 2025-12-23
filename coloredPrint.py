"""
Colored print utility for better terminal output visualization.
Supports ANSI color codes (Unix/Mac) and colorama for Windows compatibility.
"""

import sys
from datetime import datetime

# ANSI color helper function (used in both cases)
def ansi_256_fg(idx):
    """
    Returns ANSI escape code for 256-color foreground.
    idx: color index (0-255)
    """
    return f'\033[38;5;{idx}m'

# Try to import colorama for Windows support, fallback to ANSI codes
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)  # Automatically reset after each print
    USE_COLORAMA = True
    
    # Add custom 256-color support to colorama Fore (colorama only has basic 8 colors)
    Fore.ORANGE = ansi_256_fg(208)        # vivid orange
    Fore.PINK = ansi_256_fg(205)          # hot pink
    Fore.TEAL = ansi_256_fg(37)           # teal
    Fore.BROWN = ansi_256_fg(94)          # brown
    Fore.GREY = ansi_256_fg(244)          # mid grey
    Fore.LAVENDER = ansi_256_fg(183)      # lavender
    Fore.AQUA = ansi_256_fg(51)           # aqua/light blue
    Fore.CHARTREUSE = ansi_256_fg(118)    # chartreuse/lime green
    Fore.OLIVE = ansi_256_fg(100)         # olive
    Fore.MAROON = ansi_256_fg(131)        # maroon
    Fore.NAVY = ansi_256_fg(18)           # navy blue
    Fore.HOTPINK = ansi_256_fg(197)       # hot pink (stronger)
    Fore.SKY = ansi_256_fg(117)           # sky blue
    
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
        
    # INSERT_YOUR_CODE
    # There are no fundamentally different colors in the ANSI 8/16 color standard beyond the base 8 colors and their "bright" (high-intensity) versions,
    # but for some terminals, 256-color mode is available with extended, not-just-bright versions.
    # Here is an approach to allow actual different foreground colors with extended 256-color ANSI codes.



    # Some sample extra ANSI 256-color palette indexes for variety:
    Fore.ORANGE = ansi_256_fg(208)        # vivid orange
    Fore.PINK = ansi_256_fg(205)          # hot pink
    Fore.TEAL = ansi_256_fg(37)           # teal
    Fore.BROWN = ansi_256_fg(94)          # brown
    Fore.GREY = ansi_256_fg(244)          # mid grey
    Fore.LAVENDER = ansi_256_fg(183)      # lavender
    Fore.AQUA = ansi_256_fg(51)           # aqua/light blue
    Fore.CHARTREUSE = ansi_256_fg(118)    # chartreuse/lime green
    Fore.OLIVE = ansi_256_fg(100)         # olive
    Fore.MAROON = ansi_256_fg(131)        # maroon
    Fore.NAVY = ansi_256_fg(18)           # navy blue
    Fore.HOTPINK = ansi_256_fg(197)       # hot pink (stronger)
    Fore.SKY = ansi_256_fg(117)           # sky blue

    # You can add more by looking up a 256-color ANSI color chart.


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

device_colors = {
    'MEDOC': Fore.TEAL,
    'EYELINK': Fore.CYAN,
    'EEG': Fore.LAVENDER,
    'LED': Fore.SKY,
    'LOG': Fore.BROWN,
    'INFO': Fore.GREEN,
    'VIDEO CAMERA': Fore.PINK,
    'ERROR': Fore.RED,
    'WARNING': Fore.YELLOW,
    'SUCCESS': Fore.GREEN,
    'DEBUG': Fore.BLUE,
    'HIGHLIGHT': Fore.MAGENTA,
    'SESSION': Fore.CYAN,
    'TRIAL': Fore.GREEN,
    'DATA': Fore.BLUE,
}


def get_timestamp():
    """Get formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds


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
    print_with_title_color("SUCCESS", message, **kwargs)


def print_error(message, **kwargs):
    timestamp_str = f"[{get_timestamp()}] "
    colored_print(f"{timestamp_str}[ERROR] - {message}", color=Fore.RED, **kwargs)

def print_warning(message, **kwargs):
    timestamp_str = f"[{get_timestamp()}] "
    colored_print(f"{timestamp_str}[WARNING] - {message}", color=Fore.YELLOW, **kwargs)

def print_info(message, **kwargs):
    print_with_title_color("INFO", message, **kwargs)


def print_debug(message, **kwargs):
    timestamp_str = f"[{get_timestamp()}] "
    colored_print(f"{timestamp_str}[DEBUG] - {message}", color=Fore.BLUE, **kwargs)


def print_highlight(message, **kwargs):
    print_with_title_color("HIGHLIGHT", message, **kwargs)


def print_session(message, **kwargs):
    print_with_title_color("SESSION", message, **kwargs)


def print_trial(message, **kwargs):
    print_with_title_color("TRIAL", message, **kwargs)


def print_data(message, **kwargs):
    print_with_title_color("DATA", message, **kwargs)

def print_with_title_color(title, message, place_holder=False, show_timestamp=True, **kwargs):
    """
    Print a message with colored title prefix and optional timestamp.
    
    Args:
        title: Title/label for the message
        message: Message to print
        place_holder: If True, mark as placeholder (yellow color)
        show_timestamp: If True, include timestamp in output
        **kwargs: Additional arguments passed to print()
    """
    style = kwargs.get('style', Style.NORMAL)
    color = device_colors.get(title, Fore.WHITE)
    placeholder_color = Fore.YELLOW
    
    # Build prefix with timestamp if requested
    timestamp_str = f"[{get_timestamp()}] " if show_timestamp else ""
    placeholder_str = " [placeholder]" if place_holder else ""
    print(f"{timestamp_str}", end="")
    colored_print(f"[{title}]", color=color, style=style, end="")
    colored_print(placeholder_str, color=placeholder_color, style=style, end="")
    print(" - ", end="")
    print(message, **kwargs)

def logger(title, place_holder=False, show_timestamp=True):
    return lambda message, **kwargs: print_with_title_color(title, message, place_holder=place_holder, show_timestamp=show_timestamp, **kwargs)



# Example usage
if __name__ == "__main__":
    eeg_logger = logger("EEG")
    eeg_logger("This is a test message")

    medoc_logger = logger("MEDOC")
    medoc_logger("This is a test message")

    eyelink_logger = logger("EYELINK")
    eyelink_logger("This is a test message")

    led_logger = logger("LED")
    led_logger("This is a test message")

    log_logger = logger("LOG")
    log_logger("This is a test message")

    info_logger = logger("INFO")
    info_logger("This is a test message")

    print_with_title_color("MEDOC", "This is a test message", place_holder=True)
