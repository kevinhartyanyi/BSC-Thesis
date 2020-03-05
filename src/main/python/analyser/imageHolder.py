from collections import deque
import utils
from PIL import Image
import imageLoader
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from multiprocessing import Pool
import time



class imageHolder:
    def __init__(self, maxSize):
        super().__init__()
        self.images = deque(maxlen=maxSize)
        self.maxLen = maxSize
        self.vidLen = None
        self.list_idx = 0
        self.img_list = None
        self.width = 0
        self.height = 0
        self.img_dict = {}
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def setup(self, img_list, width, height):
        self.img_list = img_list
        self.vidLen = len(img_list)
        self.list_idx = 0
        self.width = width
        self.height = height
        self.img_dict.clear() #?
        self.load()

    def loadImg(self, img_path, idx):
        img = Image.open(img_path)
        img = utils.resizeImg(img, self.width)
        img = utils.fillImg(img, size=(self.width, self.height))
        print("Added: ", idx)
        self.img_dict[idx] = (img, idx)

    def load(self):        
        start = self.list_idx if self.list_idx < self.maxLen else self.list_idx - self.maxLen
        load_len = self.maxLen + start if self.maxLen + start < self.vidLen + start else self.vidLen + start
        self.list_idx = start
        print("Loading...", start, load_len)
        for i in range(start, load_len):
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], self.list_idx % self.maxLen) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1

    def reset(self):
        """
        Empty images in frameHolder.
        """
        self.images.clear()
        self.idx = -1
        self.vidLen = None
        self.img_list = None
    
    def prevImg(self):
        """
        Returns a tuple of: (previous image, True) if previous image was attainable
                            (None, False)          if it was not.
        Returns:
            (PIL Image, Boolean) -- tuple of the previous image and True, or None and False if there was no previous image
        """
        success = False
        img = None
        if self.idx > 0:
            self.idx -= 1
            img = self.images[self.idx]
            success = True
            print("PREV")
        else:
            print("NO PREV")
        print(len(self.images), self.idx)
        return img, success

    def nextImg(self, width, height):

        img = None

        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.load()
            time.sleep(1)

        cur = self.list_idx % self.maxLen

        print("Current: ", self.list_idx, cur, self.img_dict[cur][1])
        img = self.img_dict[cur][0]        

        if self.list_idx < len(self.img_list): # Load new image
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], cur) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1
        else:
            self.list_idx += 1

        return img    



