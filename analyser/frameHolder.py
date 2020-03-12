from collections import deque


class frameHolder:
    def __init__(self, maxSize):
        super().__init__()
        self.images = deque(maxlen=maxSize)
        self.maxLen = maxSize
        self.idx = -1

    def reset(self):
        """
        Empty images in frameHolder.
        """
        self.images.clear()
        self.idx = -1

    def add(self, img):
        """
        Add new image to frameHolder.
        If size is at MAX_LEN, then also remove the oldest image.
        
        Arguments:
            img {PIL Image} -- new image to be added
        """
        self.images.append(img)
        if self.idx < self.maxLen - 1:
            self.idx += 1
        print("Added")
        print(len(self.images), self.idx)
    
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
        success = False
        img = None
        if self.idx < len(self.images) - 1:
            self.idx += 1
            img = self.images[self.idx]
            success = True
            print("NEXT")
        else:
            print("NO NEXT")
        print(len(self.images), self.idx)
        return img, success
    




