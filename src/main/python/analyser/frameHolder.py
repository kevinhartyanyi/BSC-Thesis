from collections import deque


class frameHolder:
    def __init__(self, maxSize):
        super().__init__()
        self.images = deque(maxlen=maxSize)
        self.maxLen = maxSize
        self.idx = -1

    def reset(self):
        self.images.clear()
        self.idx = -1

    def add(self, img):
        self.images.append(img)
        if self.idx < self.maxLen - 1:
            self.idx += 1
        print("Added")
        print(len(self.images), self.idx)
    
    def prev(self):
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
    




