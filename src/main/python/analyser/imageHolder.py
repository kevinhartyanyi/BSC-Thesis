from collections import deque
import utils
from PIL import Image
import imageLoader
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from multiprocessing import Pool
from queue import Queue




def loadImg(img_path):
    img_p, ids, width, height = img_path
    img = Image.open(img_p)
    img = utils.resizeImg(img, width)
    img = utils.fillImg(img, size=(width, height))
    return img, ids




class imageHolder:
    def __init__(self, maxSize):
        super().__init__()
        self.images = deque(maxlen=maxSize)
        self.maxLen = maxSize
        self.vidLen = None
        self.idx = -1
        self.list_idx = 0
        self.img_list = None
        self.width = 0
        self.height = 0
        self.dict = {}
        self.que = Queue()
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def setup(self, img_list, width, height):
        self.img_list = img_list
        self.vidLen = len(img_list)
        self.list_idx = 0
        self.width = width
        self.height = height
        self.load2()

    def loadImgQt(self, img_path, ids):
        img = Image.open(img_path)
        img = utils.resizeImg(img, self.width)
        img = utils.fillImg(img, size=(self.width, self.height))
        self.que.put((img, ids))
        return img, ids

    def load2(self):
        start = self.list_idx
        self.start = start
        load_len = self.maxLen + start if self.maxLen + start < self.vidLen + start else self.vidLen + start
        #print("Start:",start, "End: ", load_len)
        
        self.images.clear()
        print("Loading...", start, load_len)
        for i in range(start, load_len):
            worker = imageLoader.Worker(self.loadImgQt, self.img_list[i], i - start) # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.added)
            self.threadpool.start(worker) 
        self.idx = 0
        print("Start2:",start)       
        
        # Execute

    def load3(self):
        start = self.list_idx
        print("Start:",start)
        load_len = self.maxLen + start if self.maxLen + start < self.vidLen + start else self.vidLen + start
        self.idx = 0
        self.images.clear()
        print("Loading...", start, load_len)

        with Pool(6) as p:
            # schedule one map/worker for each row in the original data
            q = p.map(loadImg, [(self.img_list[i], i, self.width, self.height) for i in range(start, load_len)])
        for o in q:
            print(o[1])
            self.images.append(o[0])
            self.list_idx += 1
        print("Start2:",start)
    
    def added(self, img):
        print("Added in:", img[1])
        self.dict[img[1]] = (img[0], img[1])
        self.list_idx += 1
        #self.images.append(img[0])
        
        
    
    def load(self):
        start = self.list_idx
        print("Start:",start)
        load_len = self.maxLen + start if self.maxLen + start < self.vidLen + start else self.vidLen + start
        self.idx = 0
        self.images.clear()
        print("Loading...", start, load_len)
        for i in range(start, load_len):
            img = Image.open(self.img_list[i])
            img = utils.resizeImg(img, self.width)
            img = utils.fillImg(img, size=(self.width, self.height))
            self.images.append(img)
            self.list_idx += 1
        print("Start2:",start)


    def nextI_old(self, width, height):
        """
        Returns a tuple of: (next image, True) if next image was attainable
                            (None, False)      if it was not.
        
        Returns:
            (PIL Image, Boolean) -- tuple of the next image and True, or None and False if there was no next image
        """
        img = None
        self.idx += 1

        if width != self.width or height != self.height:
            self.load3()
            self.width = width
            self.height = height

        print(len(self.images), self.idx, self.list_idx)
        if self.idx < len(self.images):  
            print("NEXT")
        else:
            print("NO NEXT => Load more")
            self.load3()

        img = self.images[self.idx]
        print("2:",len(self.images), self.idx, self.list_idx)


        return img
    

    def reset(self):
        """
        Empty images in frameHolder.
        """
        self.images.clear()
        self.idx = -1
        self.vidLen = None
        self.img_list = None
    
    def prev(self):
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

    def nextI(self, width, height):
        """
        Returns a tuple of: (next image, True) if next image was attainable
                            (None, False)      if it was not.
        
        Returns:
            (PIL Image, Boolean) -- tuple of the next image and True, or None and False if there was no next image
        """
        img = None
        self.idx += 1

        if width != self.width or height != self.height:
            self.load2()
            self.width = width
            self.height = height

        print(len(self.images), self.idx, self.list_idx)
        if self.idx < len(self.images):  
            print("NEXT")
        else:
            print("NO NEXT => Load more")
            #self.load2()

        print("Queue size: ", self.que.qsize())

        cur = self.list_idx % self.maxLen

        #print("2:",len(self.images), self.idx, self.list_idx)
        img2 = self.que.get()
        #img = img2[0]
        #print("Current: ", self.dict[self.idx][1], img2[1])
        print("Current: ", self.list_idx, cur, len(self.img_list))
        img = self.dict[cur][0]

        i = self.list_idx
        if self.list_idx < len(self.img_list):
            worker = imageLoader.Worker(self.loadImgQt, self.img_list[i], cur) # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.added)
            self.threadpool.start(worker) 
        else:
            self.list_idx += 1


        return img
    




