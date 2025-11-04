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

class Screen:
    def __init__(self, win, visuals=[]):
        self.visuals = visuals
        self.win = win

    def show(self):
        for visual in self.visulas:
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

    def show(self):
        self.win.flip()
        core.wait(0.2)

    def screen(self, name):
        self.screens[name].show()

            