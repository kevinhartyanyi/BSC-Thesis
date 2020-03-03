from PIL import Image
import glob
import cv2
import skvideo.io
import tqdm
from natsort import natsorted


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

def images_from_video(vid_path, out_path):    
    if not os.path.exists(vid_path):
        raise ValueError('Input video file %s does not exist.' % vid_path)

    cap = skvideo.io.FFmpegReader(vid_path)
    frame_nr, _, _, _ = cap.getShape()

    with tqdm.tqdm(total=frame_nr) as pbar:
        for i,frame in enumerate(cap.nextFrame()):
            Image.fromarray(frame).save(out_path + '/' + str(i) + '.png')
            pbar.update()
            
    cap.close()

def video_from_images(img_dir, out_fn, fps=30):
    onlyfiles = glob.glob(img_dir +"/*.png")
    onlyfiles = natsorted(onlyfiles)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    
    #out_fn = os.path.join(root_folder, scene_id, '{}.avi'.format(scene_id))
    img = cv2.imread(onlyfiles[0])
    #print(onlyfiles[0], img.shape)
    vid_out = cv2.VideoWriter(out_fn, fourcc, fps, (img.shape[1], img.shape[0]))

    for i in range(len(onlyfiles)):
        img = cv2.imread(onlyfiles[i])
        vid_out.write(img)
    vid_out.release()

def readImg(img_dir):
    onlyfiles = glob.glob(img_dir +"/*.png")
    onlyfiles = natsorted(onlyfiles)
    return onlyfiles

