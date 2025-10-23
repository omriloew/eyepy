import config
from pynput import mouse  # for mouse control
import random  # for random number generators
import numpy as np
from psychopy import visual, event, core, sound

def left_2_right(msg):
    if config.is_hebrew:
        return "\n".join([part[::-1] for part in msg.split('\n')])
    return msg


def generate_wait_times(mean=config.intervals_mean, std=config.intervals_std, min_time=config.intervals_min_time, max_time=config.intervals_max_time, method=config.intervals_generation_method, trails_number=config.main_trails_number):
    """
    Generate inter-trial intervals using various distribution methods.
    Different methods are used in experimental psychology for different purposes.
    
    Args:
        trails_number: Number of trials (returns trails_number wait times, last one is 0)
        mean: Mean wait time in seconds (default: 1.0)
        std: Standard deviation in seconds (default: 0.5)
        min_time: Minimum wait time in seconds (default: 0.5)
        max_time: Maximum wait time in seconds (default: 3.0)
        method: Distribution method to use (default: 'normal')
            - 'normal': Truncated normal distribution - most common, symmetric variability
            - 'exponential': Exponential distribution - models Poisson processes, more realistic for natural intervals
            - 'uniform': Uniform distribution - equal probability across range, fully unpredictable
            - 'lognormal': Log-normal distribution - right-skewed, prevents very short intervals
            - 'gamma': Gamma distribution - flexible shape, good for modeling waiting times
    
    Returns:
        List of wait times in seconds (length = trails_number, last element = 0)
    """
    wait_times = []
    # Generate n-1 intervals (between trials)
    for _ in range(trails_number):
        if method == 'normal':
            # Normal distribution - symmetric around mean
            wait_time = np.random.normal(mean, std)
            wait_time = np.clip(wait_time, min_time, max_time)
            
        elif method == 'exponential':
            # Exponential distribution - models Poisson process
            # Rate parameter is 1/mean
            wait_time = np.random.exponential(mean)
            wait_time = np.clip(wait_time, min_time, max_time)
            
        elif method == 'uniform':
            # Uniform distribution - completely random within bounds
            wait_time = np.random.uniform(min_time, max_time)
            
        elif method == 'lognormal':
            # Log-normal distribution - right-skewed, avoids very short intervals
            # Calculate mu and sigma for desired mean and std
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            wait_time = np.random.lognormal(mu, sigma)
            wait_time = np.clip(wait_time, min_time, max_time)
            
        elif method == 'gamma':
            # Gamma distribution - flexible shape parameter
            # Shape parameter k and scale parameter theta
            k = (mean / std) ** 2
            theta = (std ** 2) / mean
            wait_time = np.random.gamma(k, theta)
            wait_time = np.clip(wait_time, min_time, max_time)
        
        elif method == 'fixed':
            wait_time = mean
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: 'normal', 'exponential', 'uniform', 'lognormal', 'gamma', 'fixed'")
        
        wait_times.append(wait_time)
    
    
    return wait_times

def pain_vas(win, timeout=config.vas_timeout, instructions=None):
    """
    Visual Analog Scale for pain rating using PsychoPy Slider.
    
    Args:
        win: PsychoPy window object
        timeout: Optional timeout in seconds (None for no timeout)
        instructions: Custom instruction text to display
    
    Returns:
        tuple: (rating_value, reaction_time)
    """
    # Make mouse visible for dragging
    win.mouseVisible = True
    
    # Create instruction text - ABOVE the scale - spread across screen
    instruction_text = visual.TextStim(
        win,
        text=instructions,
        color=config.txt_color,
        height=config.txt_size * 1.1,  # Just slightly larger than normal
        alignText='center',
        wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1  # Allow text to spread across screen
    )
    instruction_text.pos = [0, config.scrsize[1]*0.3]  
    
    # Create the VAS slider - configurable via config
    vas_slider = visual.Slider(
        win=win,
        size=(1.2, 0.1),
        pos=(0, 0),
        units='norm',
        labels=['No Pain', 'Worst Pain Imaginable'],
        ticks=(config.vas_min, config.vas_max),
        granularity=config.vas_granularity,
        startValue=config.vas_start_value,
        color=config.txt_color,
        markerColor=config.vas_marker_color,
        style=config.vas_marker_style,
    )
    
    # Create current value text - shows numeric value if enabled
    value_text = None
    if config.vas_show_value:
        value_text = visual.TextStim(
            win,
            text=str(config.vas_start_value),
            color=config.txt_color,
            height=config.txt_size,
            alignText='center',
            bold=True
        )
        value_text.pos = [0, config.scrsize[1]/2*0.15]  # Between instruction and slider
    
    # Create control instruction - BELOW the scale - smaller and spread
    control_text = visual.TextStim(
        win,
        text="Click and drag OR use LEFT/RIGHT arrows to move, SPACE to confirm.",
        color=config.txt_color,
        height=config.txt_size * 0.7,  # Much smaller
        alignText='center',
        wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1  # Allow text to spread across screen
    )
    control_text.pos = [0, -config.scrsize[1]/2*0.3]  # Closer to the scale
    
    # Start clock for reaction time
    start_time = core.monotonicClock.getTime()
    event.clearEvents()
    
    # Main VAS loop
    while True:
        # Get current rating for display
        current_rating = vas_slider.getRating()
        if current_rating is None:
            current_rating = config.vas_start_value
        
        # Update value text if enabled
        if config.vas_show_value and value_text is not None:
            value_text.text = f"{int(current_rating)}"
        
        # Draw all elements
        instruction_text.draw()
        vas_slider.draw()
        control_text.draw()
        
        # Draw value text if enabled
        if config.vas_show_value and value_text is not None:
            value_text.draw()
        
        win.flip()
        
        # Small wait to prevent CPU spinning
        core.wait(0.001)
        
        # Check for keyboard input
        keys = event.getKeys(keyList=['left', 'right', 'space', 'return', 'escape'])
        
        if keys:
            if 'left' in keys:
                # Move left (decrease rating)
                current_value = vas_slider.getRating()
                if current_value is None:
                    current_value = config.vas_start_value
                new_value = max(config.vas_min, current_value - config.vas_keyboard_step)
                vas_slider.setRating(new_value)
            elif 'right' in keys:
                # Move right (increase rating)
                current_value = vas_slider.getRating()
                if current_value is None:
                    current_value = config.vas_start_value
                new_value = min(config.vas_max, current_value + config.vas_keyboard_step)
                vas_slider.setRating(new_value)
            elif 'space' in keys or 'return' in keys:
                # Confirm selection
                break
            elif 'escape' in keys:
                # Exit without response
                return None, core.monotonicClock.getTime() - start_time
        
        # Check for timeout
        if timeout is not None:
            if core.monotonicClock.getTime() - start_time > timeout:
                return None, core.monotonicClock.getTime() - start_time
    
    # Get final rating and reaction time
    final_rating = vas_slider.getRating()
    
    # If rating is None (slider never moved), use the start value
    if final_rating is None:
        final_rating = config.vas_start_value
    
    reaction_time = core.monotonicClock.getTime() - start_time
    
    # Hide mouse again
    win.mouseVisible = False
    
    return final_rating, reaction_time