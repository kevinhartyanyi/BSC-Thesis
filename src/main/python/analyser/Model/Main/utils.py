from PIL import Image
import glob
import cv2
import skvideo.io
import tqdm
from natsort import natsorted


def resizeImg(img, new_width, new_height):
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
    pwidth = new_width / float(w)
    height = int((float(h) * float(pwidth)))
    if height > new_height:
        height = new_height
        pheight = height / float(h)
        width = int((float(w) * float(pheight)))
    img = img.resize((width, height), Image.ANTIALIAS)
    return img


def fillImg(img, fill_colour=(26, 26, 27, 255), size=(1920, 1080)):
    """
    Creates new image with fill_colour and pastes the given onto it, returned image is size sized.
    
    Arguments:
        img {PIL Image} -- Image to be filled
    
    Keyword Arguments:
        fill_colour {tuple} -- filling colour for the image (default: {(26,26,27,255)})
        size {tuple} -- new size for the image (default: {(1920, 1080)})
    
    Returns:
        PIL Image -- resized and filled image
    """
    w, h = img.size
    if len(fill_colour) == 3:
        A = 255
        fill_colour = fill_colour + (A,)
    fd_img = Image.new("RGBA", size, fill_colour)
    fd_img.paste(img, ((int((size[0] - w) / 2), int((size[1] - h) / 2))))
    return fd_img


def imagesFromVideo(vid_path, out_path):
    """Saves every frame of a video to the given path.
    
    Arguments:
        vid_path {str} -- path to video
        out_path {str} -- path to save dir
    
    Raises:
        ValueError: If video doesn't exist
    """
    if not os.path.exists(vid_path):
        raise ValueError("Input video file %s does not exist." % vid_path)

    cap = skvideo.io.FFmpegReader(vid_path)
    frame_nr, _, _, _ = cap.getShape()

    with tqdm.tqdm(total=frame_nr) as pbar:
        for i, frame in enumerate(cap.nextFrame()):
            Image.fromarray(frame).save(out_path + "/" + str(i) + ".png")
            pbar.update()

    cap.close()


def videoFromImages(img_dir, out_dir, fps=30):
    """Creates a video based on the png files contained in the given path. Saves the video to out_dir.
    
    Arguments:
        img_dir {str} -- path to image dir
        out_dir {str} -- path to save dir
    
    Keyword Arguments:
        fps {int} -- the fps of the video (default: {30})
    """
    onlyfiles = glob.glob(img_dir + "/*.png")
    onlyfiles = natsorted(onlyfiles)

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

    img = cv2.imread(onlyfiles[0])
    vid_out = cv2.VideoWriter(out_dir, fourcc, fps, (img.shape[1], img.shape[0]))

    for i in range(len(onlyfiles)):
        img = cv2.imread(onlyfiles[i])
        vid_out.write(img)
    vid_out.release()
