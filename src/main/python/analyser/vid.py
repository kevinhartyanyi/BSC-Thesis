
class vidData:
    def __init__(self):
        super().__init__()
        self.current_idx = None
        self.end_idx = None
        self.fps = None
        self.end_idx = None
        self.height = None
        self.width = None
        self.channels = None

    def setup(self, vid_shape):
        frames, height, width, channels = vid_shape
        self.height = height
        self.width = width
        self.channels = channels

        self.current_idx = 0
        self.end_idx = frames
        self.fps = 30
    
    def getStartData(self):
        return self.current_idx, self.end_idx, self.fps
