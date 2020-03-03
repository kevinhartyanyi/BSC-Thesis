from collections import deque
import utils
from PIL import Image

class imageHolder:
    def __init__(self, maxSize):
        super().__init__()
        self.images = deque(maxlen=maxSize)
        self.maxLen = maxSize
        self.vidLen = None
        self.idx = -1
        self.list_idx = 0
        self.img_list = None

    def setup(self, img_list):
        self.img_list = img_list
        self.vidLen = len(img_list)
        self.list_idx = 0
        self.load()
        
    
    def load(self):
        start = self.list_idx
        load_len = self.maxLen + start if self.maxLen + start < self.vidLen + start else self.vidLen + start
        self.idx = -1
        self.images.clear()
        print("Loading...", start, load_len)
        for i in range(start, load_len):
            self.images.append(Image.open(self.img_list[i]))
            self.list_idx += 1

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

    def nextI(self):
        """
        Returns a tuple of: (next image, True) if next image was attainable
                            (None, False)      if it was not.
        
        Returns:
            (PIL Image, Boolean) -- tuple of the next image and True, or None and False if there was no next image
        """
        print(len(self.images), self.idx)
        img = None
        self.idx += 1
        if self.idx < len(self.images) - 1:  
            print("NEXT")
        else:
            print("NO NEXT => Load more")
            self.load()

        img = self.images[self.idx]

        return img
    




