from psychopy import visual, core
import config

def create_instructions(win, instructions):
    return visual.TextStim(win, text=instructions, color=config.txt_color, height=config.txt_size,
        alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1)

def top_instructions(win, instructions):
    return visual.TextStim(win, text=instructions, color=config.txt_color, height=config.txt_size,
        alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
        pos=(0, config.scrsize[1]*0.3))

def middle_instructions(win, instructions):
    return visual.TextStim(win, text=instructions, color=config.txt_color, height=config.txt_size,
        alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
        pos=(0, 0))

def bottom_instructions(win, instructions):
    return visual.TextStim(win, text=instructions, color=config.txt_color, height=config.txt_size,
        alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
        pos=(0, -config.scrsize[1]*0.3))

def fixation_cross(win):
    return visual.TextStim(win, text='+', color=config.txt_color, height=150,
        alignText='center', antialias=False)

def blank(win):
    return visual.TextStim(win, text=' ', color=config.txt_color, height=150,
        alignText='center', antialias=False)

def create_vas_scale(win, percentage=50, left_label='0', right_label='100', 
                     scale_width=None, scale_height=20, tick_length=10, 
                     cursor_size=15, position='top', cursor_color='red'):
    """
    Create a VAS (Visual Analog Scale) at the top of the screen with labels, tick marks, and a cursor.
    
    Args:
        win: PsychoPy window object
        percentage: Position of cursor as percentage from left (0-100)
        left_label: Label text for left side (default: '0')
        right_label: Label text for right side (default: '100')
        scale_width: Width of the scale in pixels (default: 80% of screen width)
        scale_height: Height of the main scale line in pixels
        tick_length: Length of tick marks in pixels
        cursor_size: Size of the cursor/pointer in pixels
        position: Vertical position ('top', 'center', 'bottom', or custom y value)
        cursor_color: Color of the cursor (default: 'red')
    
    Returns:
        List of visual elements that can be drawn
    """
    if scale_width is None:
        scale_width = config.scrsize[0] * 0.8
    
    # Determine vertical position
    if position == 'top':
        y_pos = config.scrsize[1] * 0.4
    elif position == 'center':
        y_pos = 0
    elif position == 'bottom':
        y_pos = -config.scrsize[1] * 0.4
    else:
        y_pos = position  # Assume it's a numeric value
    
    visuals = []
    
    # Main horizontal line
    main_line = visual.Line(
        win,
        start=(-scale_width/2, y_pos),
        end=(scale_width/2, y_pos),
        lineColor=config.txt_color,
        lineWidth=2
    )
    visuals.append(main_line)
    
    # Calculate positions for tick marks (at 0, 25, 50, 75, 100%)
    tick_positions = [0, 25, 50, 75, 100]
    for tick_percent in tick_positions:
        x_pos = -scale_width/2 + (scale_width * tick_percent / 100)
        
        # Tick mark line (vertical)
        tick_line = visual.Line(
            win,
            start=(x_pos, y_pos - tick_length/2),
            end=(x_pos, y_pos + tick_length/2),
            lineColor=config.txt_color,
            lineWidth=1
        )
        visuals.append(tick_line)
    
    # Left label
    left_label_text = visual.TextStim(
        win,
        text=left_label,
        color=config.txt_color,
        height=config.txt_size * 0.8,
        alignText='center',
        pos=(-scale_width/2 - 50, y_pos)
    )
    visuals.append(left_label_text)
    
    # Right label
    right_label_text = visual.TextStim(
        win,
        text=right_label,
        color=config.txt_color,
        height=config.txt_size * 0.8,
        alignText='center',
        pos=(scale_width/2 + 50, y_pos)
    )
    visuals.append(right_label_text)
    
    # Cursor/pointer at the specified percentage
    cursor_x = -scale_width/2 + (scale_width * percentage / 100)
    
    # Draw cursor as a triangle pointing down
    cursor = visual.Polygon(
        win,
        edges=3,
        radius=cursor_size,
        pos=(cursor_x, y_pos - cursor_size),
        ori=180,  # Rotate so triangle points down
        fillColor=cursor_color,
        lineColor=cursor_color,
        lineWidth=2
    )
    visuals.append(cursor)
    
    # Optional: Add a vertical line from cursor to scale
    cursor_line = visual.Line(
        win,
        start=(cursor_x, y_pos),
        end=(cursor_x, y_pos - cursor_size),
        lineColor=cursor_color,
        lineWidth=2
    )
    visuals.append(cursor_line)
    
    return visuals

class Screen:
    def __init__(self, win, visuals=[]):
        self.visuals = visuals
        self.win = win

    def show(self):
        for visual in self.visuals:
            visual.draw()
        self.win.flip()
        core.wait(0.2)

    def add_visual(self, visual):
        self.visuals.append(visual)

    def add_screen(self, screen):
        self.screens[screen.name] = screen


class Draw:

    def __init__(self, win):
        self.win = win
        self.screens = {}
        self.vas_scale_visuals = None  # Store VAS scale visuals for repeated drawing

    
    
    def instructions(self, instructions):
        create_instructions(self.win, instructions).draw()

    def top_instructions(self, instructions):
        top_instructions(self.win, instructions).draw()

    def middle_instructions(self, instructions):
        middle_instructions(self.win, instructions).draw()

    def bottom_instructions(self, instructions):
        bottom_instructions(self.win, instructions).draw()

    def fixation_cross(self):
        fixation_cross(self.win).draw()

    def blank(self):
        blank(self.win).draw()

    def vas_scale(self, percentage=50, left_label='0', right_label='100', 
                  scale_width=config.scrsize[0]*0.8, tick_length=10, cursor_size=15, 
                  position='top', cursor_color='red', store=True):
        """
        Draw a VAS scale at the top of the screen.
        
        Args:
            percentage: Position of cursor as percentage from left (0-100)
            left_label: Label text for left side (default: '0')
            right_label: Label text for right side (default: '100')
            scale_width: Width of the scale in pixels (default: 80% of screen width)
            tick_length: Length of tick marks in pixels
            cursor_size: Size of the cursor/pointer in pixels
            position: Vertical position ('top', 'center', 'bottom', or custom y value)
            cursor_color: Color of the cursor (default: 'red')
            store: If True, store the visuals for repeated drawing (default: True)
        """
        # Create the scale visuals
        visuals = create_vas_scale(
            self.win, 
            percentage=percentage,
            left_label=left_label,
            right_label=right_label,
            scale_width=scale_width,
            tick_length=tick_length,
            cursor_size=cursor_size,
            position=position,
            cursor_color=cursor_color
        )
        
        # Store visuals if requested
        if store:
            self.vas_scale_visuals = visuals
        
        # Draw all visuals
        for visual_element in visuals:
            visual_element.draw()
    
    def draw_stored_vas_scale(self):
        """Draw the stored VAS scale (if it exists)."""
        if self.vas_scale_visuals is not None:
            for visual_element in self.vas_scale_visuals:
                visual_element.draw()

    def show(self):
        self.win.flip()
        core.wait(0.2)

    def screen(self, name):
        self.screens[name].show()

            