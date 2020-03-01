from PIL import Image

def resizeImg(img, new_width):
    """
    Resize image while keeping aspect ratio
    
    Arguments:
        img {PIL image} -- image to be resized
        new_width {int} -- width for the new image
    
    Returns:
        PIL image -- the resized image
    """
    w, h = img.size
    width = new_width
    pwidth = (new_width/float(w))
    height = int((float(h)*float(pwidth)))
    #image.resize((width,hsize), Image.BICUBIC).show()
    img = img.resize((width,height), Image.ANTIALIAS)
    return img

def fillImg(img, fill_color=(26,26,27,255), size=(1920, 1080)):
    """
    Creates new image with fill_color and pastes the given onto it, returned image is size sized.
    
    Arguments:
        img {[type]} -- [description]
    
    Keyword Arguments:
        fill_color {tuple} -- [description] (default: {(26,26,27,255)})
        size {tuple} -- [description] (default: {(1920, 1080)})
    
    Returns:
        [type] -- [description]
    """
    w, h = img.size
    fd_img = Image.new('RGBA', size, fill_color)
    fd_img.paste(img, ((int((size[0] - w) / 2), int((size[1] - h) / 2))))
    return fd_img
