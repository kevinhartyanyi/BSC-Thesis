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
        self.maxLen = maxSize
        self.vidLen = None
        self.img_list = None
        self.current = None
        self.list_idx = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.cur_idx = -1
        self.img_dict = {}
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def setup(self, img_list, width, height, fps, n_frame=None):
        self.img_list = img_list
        self.vidLen = len(img_list)
        self.list_idx = 0
        self.fps = fps
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
        img = None
        if self.cur_idx > -1:
            img = Image.open(self.img_list[self.cur_idx])
        else:
            img = Image.open(self.img_list[0])
        
        return img

    def increment(self):
        self.cur_idx += 1
        self.list_idx += 1

    
    def getStartData(self):
        """
        Returns the current index, end index and fps in a tuple.
        This function is used for initializing the worker in startVideo.
        
        Returns:
            (int, int, int) -- current index, end index and fps
        """
        return self.cur_idx, self.vidLen, self.fps

    def jump(self, n_frame):
        self.load(begin=n_frame)
        self.cur_idx = n_frame
        self.list_idx = self.cur_idx + self.maxLen + 1
        return self.prepareImg(Image.open(self.img_list[n_frame]))

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.load()
        return self.prepareImg(self.getCurrent())
    
    def prepareImg(self, img):
        img = utils.resizeImg(img, self.width)
        img = utils.fillImg(img, size=(self.width, self.height))
        return img

    def loadImg(self, img_path, idx):
        img = Image.open(img_path)
        img = self.prepareImg(img)
        print("Added: ", idx)
        self.img_dict[idx] = (img, idx)

    def load(self, begin=None, end=None):
        if begin == None:        
            start = self.list_idx if self.list_idx < self.maxLen else self.list_idx - self.maxLen
            load_len = self.maxLen + start if self.maxLen + start < self.vidLen else self.vidLen
        elif begin != None:
            start = begin
            #if end > self.vidLen:
            #    self.list_idx = start + self.maxLen + 1
            #    return
            if self.list_idx >= self.vidLen:
                self.list_idx = start + self.maxLen + 1
                return
            load_len = start + self.maxLen + 1 if start + self.maxLen + 1 < self.vidLen else self.vidLen
        self.list_idx = start
        img = Image.open(self.img_list[start])
        img = self.prepareImg(img)
        self.current = img
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
        self.cur_idx -= 1
        print("Prev: ", self.cur_idx)
        img = Image.open(self.img_list[self.cur_idx])
        self.load(self.cur_idx, self.cur_idx + self.maxLen + 1)
        img = self.prepareImg(img)

        return img  

    def nextImg(self):
        self.cur_idx += 1
        cur = self.list_idx % self.maxLen
        if cur not in self.img_dict:
            print("Warning: Not in dictionary. Returning previous image")
            return self.current
        print("Current: ", self.list_idx, cur, self.img_dict[cur][1])
        img = self.img_dict[cur][0]    
        self.current = img    

        if self.list_idx < self.vidLen: # Load new image
            worker = imageLoader.Worker(self.loadImg, self.img_list[self.list_idx], cur) # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker) 
            self.list_idx += 1
        else:
            self.list_idx += 1

        return img    



