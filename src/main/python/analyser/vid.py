from PIL import Image

class vidData:
    def __init__(self):
        super().__init__()
        self.current_idx = None
        self.end_idx = None
        self.height = None
        self.width = None
        self.channels = None
        self.fps = 30
    
    def setupWithList(self, img_list):
        self.height = height
        self.width = width
        print(Image.open(img_list[0]).size)

        self.current_idx = 0
        self.end_idx = len(img_list) 

    def setup(self, vid_shape):
        """
        Load parameters from the video
        
        Arguments:
            vid_shape {tuple} -- frames, height, width, channels
        """
        frames, height, width, channels = vid_shape
        self.height = height
        self.width = width
        self.channels = channels

        self.current_idx = 0
        self.end_idx = frames        
    
    def getStartData(self):
        """
        Returns the current index, end index and fps in a tuple.
        This function is used for initializing the worker in startVideo.
        
        Returns:
            [type] -- [description]
        """
        return self.current_idx, self.end_idx, self.fps
