import utils
from PIL import Image
import imageLoader
from PyQt5.QtCore import QThread, QThreadPool
from multiprocessing import Pool
import logging

class imageHolder:

    def __init__(self, maxSize, fps):
        super().__init__()
        self.maxLen = maxSize
        self.vidLen = None
        self.img_list = None
        self.current = None
        self.list_idx = 0
        self.fps = fps
        self.width = 0
        self.height = 0
        self.cur_idx = -1
        self.img_dict = {}
        self.threadpool = QThreadPool()
        logging.info("Multithreading with maximum {0} threads".format(self.threadpool.maxThreadCount()))

    def setup(self, img_list, width, height, colour, n_frame=None):
        """Setup for the class to function properly
        
        Arguments:
            img_list {[str]} -- list of paths to the individual images
            width {int} -- the width of the image to be resized after loaded
            height {int} -- the height of the image to be resized after loaded
            colour {str} -- hex code of colour to fill the empty parts of the image
        
        Keyword Arguments:
            n_frame {int} -- if given then the load starts from this frame (default: {None})
        
        Returns:
            PIL Image -- the current image after loaded and resized
        """
        self.img_list = img_list
        self.vidLen = len(img_list)
        self.list_idx = 0
        colour = colour.lstrip('#')
        self.colour = tuple(int(colour[i:i+2], 16) for i in (0, 2, 4))
        self.cur_idx = -1
        self.width = width
        self.height = height
        self.img_dict.clear() #?

        img = None
        if n_frame == None:
            img = self.prepareImg(Image.open(self.img_list[0]))
            self.list_idx += 1
            self.cur_idx += 1
            self.load()
        else:
            if n_frame > self.vidLen - 1:
                n_frame = self.vidLen - 1
            img = self.jump(n_frame)
        return img

    def getCurrent(self):
        """Returns the current image
        
        Returns:
            PIL Image -- the current image
        """
        img = None
        if self.cur_idx > -1:
            img = Image.open(self.img_list[self.cur_idx])
        else:
            img = Image.open(self.img_list[0])
        
        return img

    def increment(self):
        """Increments the counters for the cur_idx and list_idx
        """
        self.cur_idx += 1
        self.list_idx += 1

    def changeFps(self, fps):
        """Change fps to the given value and store it
        
        Arguments:
            fps {int} -- frame per second
        """
        self.fps = fps

    def getStartData(self):
        """
        Returns the current index, end index and fps in a tuple.
        This function is used for initializing the worker in startVideo.
        
        Returns:
            (int, int, int) -- Tuple: current index, end index and fps
        """
        return self.cur_idx, self.vidLen, self.fps

    def jump(self, n_frame):
        """Jump to the given frame
        
        Arguments:
            n_frame {int} -- the frame number to jump to
        
        Returns:
            PIL Image -- the image at the frame number
        """
        cur_idx = n_frame
        if n_frame >= self.vidLen:
            n_frame = self.vidLen - 1
        self.load(begin=n_frame)
        self.cur_idx = cur_idx
        self.list_idx = self.cur_idx + self.maxLen + 1
        return self.prepareImg(Image.open(self.img_list[n_frame]))

    def resize(self, width, height):
        """Change resize values for images
        
        Arguments:
            width {int} -- width of the image
            height {int} -- height of the image
        
        Returns:
            PIL Image -- the current image resized with the new values
        """
        self.width = width
        self.height = height
        self.load()
        return self.prepareImg(self.getCurrent())
    
    def prepareImg(self, img):
        """Resize image keeping the aspect ratio 
        and fill the empty space (if any) between the desired size and the resized image
        
        Arguments:
            img {PIL Image} -- the image to be resized
        
        Returns:
            PIL Image -- the resized image
        """
        img = utils.resizeImg(img, self.width, self.height)
        img = utils.fillImg(img, fill_colour=self.colour, size=(self.width, self.height))
        return img

    def loadImg(self, img_path, idx):
        """Load the image at the img_path
        
        Arguments:
            img_path {str} -- path to the image
            idx {int} -- the index of the image
        """
        img = Image.open(img_path)
        img = self.prepareImg(img)
        logging.info("Added: {0}".format(idx))
        self.img_dict[idx] = (img, idx)

    def load(self, begin=None, end=None):
        """Load in images in separate threads for efficiency
        
        Keyword Arguments:
            begin {int} -- from where the load should start (default: {None})
            end {int} -- where the load should end (default: {None})
        """
        self.img_dict.clear()
        if begin == None:        
            start = self.list_idx if self.list_idx < self.maxLen else self.list_idx - self.maxLen
            load_len = self.maxLen + start if self.maxLen + start < self.vidLen else self.vidLen
        elif begin != None:
            start = begin
            if self.list_idx >= self.vidLen:
                logging.info("Skip Load {0} {1} {2}".format(self.list_idx, self.vidLen, start))
                
            load_len = start + self.maxLen + 1 if start + self.maxLen + 1 < self.vidLen else self.vidLen
        self.list_idx = start
        img = Image.open(self.img_list[start])
        img = self.prepareImg(img)
        self.current = img
        logging.info("Loading... {0} {1}".format(start, load_len))
        for i in range(start, load_len):
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], self.list_idx % self.maxLen) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1

    def reset(self):
        """
        Clear images
        """
        self.images.clear()
        self.idx = -1
        self.vidLen = None
        self.img_list = None
    
    def prevImg(self):
        """Returns the previous image from the image list
        
        Returns:
            PIL Image -- the previous image
        """
        self.cur_idx -= 1
        if self.cur_idx >= self.vidLen:
            return self.current
        logging.info("Prev: {0}".format(self.cur_idx))
        img = Image.open(self.img_list[self.cur_idx])
        self.load(self.cur_idx, self.cur_idx + self.maxLen + 1)
        img = self.prepareImg(img)

        return img  

    def nextImg(self):
        """Returns the next image from the image list
        
        Returns:
            PIL Image -- the next image
        """
        self.cur_idx += 1
        cur = self.cur_idx % self.maxLen
        if self.cur_idx >= self.vidLen:
            return self.current
        elif cur not in self.img_dict:
            logging.warning("Not in dictionary. Returning previous image")
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], cur) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1
            return self.current
        
        logging.info("Current: {0} {1} {2} {3}".format(self.list_idx, cur, self.img_dict[cur][1], self.cur_idx))
        img = self.img_dict[cur][0]    
        self.current = img    

        if self.list_idx < self.vidLen: # Load new image
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], cur) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1
        return img    



