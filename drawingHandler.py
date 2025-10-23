from psychopy import visual, core
import config

class Draw:

    def __init__(self, win):
        self.win = win
    
    def instructions(self, instructions):
           visual.TextStim(self.win, text=instructions, color=config.txt_color, height=config.txt_size,
        alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1).draw()

    def top_instructions(self, instructions):
        visual.TextStim(self.win, text=instructions, color=config.txt_color, height=config.txt_size,
            alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
            pos=(0, config.scrsize[1]*0.3)).draw()

    def middle_instructions(self, instructions):
        visual.TextStim(self.win, text=instructions, color=config.txt_color, height=config.txt_size,
            alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
            pos=(0, 0)).draw()

    def bottom_instructions(self, instructions):
        visual.TextStim(self.win, text=instructions, color=config.txt_color, height=config.txt_size,
            alignText='center', antialias=False, wrapWidth=config.scrsize[0]-config.scrsize[0]*0.1,
            pos=(0, -config.scrsize[1]*0.3)).draw()

    def fixation_cross(self):
        visual.TextStim(self.win, text='+', color=config.txt_color, height=150,
            alignText='center', antialias=False).draw()

    def blank(self):
        blank = visual.TextStim(self.win, text=' ', color=config.txt_color, height=150,
            alignText='center', antialias=False)
        blank.draw()

    def show(self):
        self.win.flip()
        core.wait(0.2)

    
            