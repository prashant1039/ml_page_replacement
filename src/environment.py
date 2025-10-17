import numpy as np

class PageReplacementEnv:
    def __init__(self, num_frames, page_sequence):
        self.num_frames = num_frames
        self.page_sequence = page_sequence
        self.frames = []
        self.pointer = 0
        self.page_faults = 0

    def reset(self):
        self.frames = []
        self.pointer = 0
        self.page_faults = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.num_frames)
        for i, page in enumerate(self.frames):
            state[i] = page
        return state

    def step(self, action):
        done = False
        reward = 0
        current_page = self.page_sequence[self.pointer]

        if current_page in self.frames:
            reward = 1
        else:
            self.page_faults += 1
            reward = -1
            if len(self.frames) < self.num_frames:
                self.frames.append(current_page)
            else:
                self.frames[action] = current_page

        self.pointer += 1
        if self.pointer >= len(self.page_sequence):
            done = True

        return self.get_state(), reward, done