
class cycleVid:
    def __init__(self):
        super().__init__()
        self.container = []
        self.idx = 0

    def reset(self):
        """
        Empty images in frameHolder.
        """
        self.idx = 0
        self.container = []

    def add(self, name, vid_path):
        self.container.append((name, vid_path))

    def current(self):
        return self.container[self.idx][1]
    
    def down(self):
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = len(self.container) - 1
            
        return self.container[self.idx][1]

    def up(self):
        if self.idx < len(self.container) - 1:
            self.idx += 1
        else:
            self.idx = 0
            
        return self.container[self.idx][1]
    




