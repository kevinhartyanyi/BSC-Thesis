import logging

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

    def get(self, name):
        """Finds and returns the video with the given name
        
        Arguments:
            name {str} -- the name of the video
        
        Returns:
            str -- path to the folder where the video is
        """
        re = None
        logging.info("L: {0}".format(len(self.container)))
        self.idx = 0
        for n, vid in self.container:
            if n == name:
                re = vid
                break
            self.idx += 1
        assert re != None, ("Could not find cycleVid with name {0}".format(name))
        
        return re

    def add(self, name, vid_path):
        """Adds a new video to the container
        
        Arguments:
            name {str} -- name of the video
            vid_path {str} -- path to the video
        """
        self.container.append((name, vid_path))
        print("Add", name)

    def current(self):
        """Returns the path to the currently selected video
        
        Returns:
            str -- path to the currently selected video
        """
        return self.container[self.idx][1]
    
    def currentType(self):
        """Returns the name of the currently selected video
        
        Returns:
            str -- name of the currently selected video
        """
        return self.container[self.idx][0]
    
    def down(self):
        """Cycle down on the container and return the path to the previous video
        
        Returns:
            str -- path to the previous video
        """
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = len(self.container) - 1
            
        logging.info("Cycle: {0}".format(self.container[self.idx][0]))
        return self.container[self.idx][1]

    def up(self):
        """Cycle up on the container and return the path to the next video
        
        Returns:
            str -- path to the next video
        """
        if self.idx < len(self.container) - 1:
            self.idx += 1
        else:
            self.idx = 0
            
        logging.info("Cycle: {0}".format(self.container[self.idx][0]))
        logging.info("Cycle: {0}".format(self.idx, len(self.container)))
        return self.container[self.idx][1]
    




