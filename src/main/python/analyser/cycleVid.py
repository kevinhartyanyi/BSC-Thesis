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
        self.container.append((name, vid_path))
        print("Add", name)

    def current(self):
        print(len(self.container))
        print(self.idx)
        return self.container[self.idx][1]
    
    def currentType(self):
        return self.container[self.idx][0]
    
    def down(self):
        if self.idx > 0:
            self.idx -= 1
        else:
            self.idx = len(self.container) - 1
            
        logging.info("Cycle: {0}".format(self.container[self.idx][0]))
        return self.container[self.idx][1]

    def up(self):
        if self.idx < len(self.container) - 1:
            self.idx += 1
        else:
            self.idx = 0
            
        logging.info("Cycle: {0}".format(self.container[self.idx][0]))
        logging.info("Cycle: {0}".format(self.idx, len(self.container)))
        return self.container[self.idx][1]
    




