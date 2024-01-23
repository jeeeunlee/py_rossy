import cv2
import numpy as np

def image_control(image, 
                  alpha = 1.0, #  Simple contrast control, 
                  beta = 0.0, # Simple brightness control
                  ):
    # ret = a * img + b
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# morphological transformations
def clear_by_opening(image):
    # removing outside noise
    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=3)
    return opening

def clear_by_closing(image):
    # filling inside noise
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=5)
    return closing

def edge_clear(mask):
    height,width = mask.shape
    skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    for i in range(5):
        eroded = cv2.erode(mask,kernel)  
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(mask,temp)
        skel = cv2.bitwise_or(skel,temp)
        mask = eroded.copy()
    return skel

def white_color_filter(img):
    # white color mask
    converted = 255 - img
    image = cv2.cvtColor(converted,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 100, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # green color mask
    return white_mask

def blue_color_filter(rgb):
    # blue color mask in rgb
    lower0 = np.uint8([100, 0, 0])
    upper0 = np.uint8([255, 150, 150])
    blue_mask0 = cv2.inRange(rgb, lower0, upper0)
    # blue_mask0 = cv2.bitwise_not(blue_mask0)
    img = cv2.bitwise_or(rgb, rgb, mask=blue_mask0)
    cv2.imshow('blue_mask0',blue_mask0)
    cv2.imshow('img',img)

    # blue color mask in hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)
    lower = np.uint8([100, 100, 0])
    upper = np.uint8([212, 255, 255])
    blue_mask = cv2.inRange(hsv, lower, upper)
    # green color mask
    return blue_mask

def image_filter(img):
    # white and green filtered
    # white color mask
    converted = 255 - img
    image = cv2.cvtColor(converted,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 100, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # green color mask
    ## Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.uint8([30, 25, 0])
    upper = np.uint8([150, 255,255])
    green_mask = cv2.inRange(hsv, lower, upper)
    green_out = cv2.bitwise_not(green_mask)

    # combine the mask
    mask = cv2.bitwise_and(white_mask, green_out)
    mask = cv2.bitwise_or(mask, mask, image)
    mask0 = mask.copy()

    # yellow mask
    # lower = np.uint8([190, 190, 0])
    # upper = np.uint8([255, 255,255])
    # yellow_mask = cv2.inRange(hsv, lower, upper)
    # mask = cv2.bitwise_or(white_mask, yellow_mask, image)

    height,width = mask.shape
    skel = np.zeros([height,width],dtype=np.uint8)      #[height,width,3]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    for i in range(5):
        eroded = cv2.erode(mask,kernel)  
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(mask,temp)
        skel = cv2.bitwise_or(skel,temp)
        mask = eroded.copy()

    return mask0, skel


def image_filter2(img):
    # thresholding
    # blur = cv2.GaussianBlur(img, (3,3), 1)
    med = np.array([75, 70, 75])
    rng = np.array([30, 30, 30])
    low = med - rng
    high = med + rng
    thresh = cv2.inRange(img, low, high)

    # morphological operations to get the paper
    kclose = np.ones((3,3), dtype=np.uint8)
    kopen = np.ones((3,3), dtype=np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kclose, iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kopen, iterations=6)
    lineleft = thresh + cv2.bitwise_not(opening)
    lineleft = cv2.GaussianBlur(lineleft, (3,3), 1)
    return lineleft, thresh, closing, opening

def image_processing(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered,_,_,_ = image_filter2(image)

    # processed = (thresh*10/21. + gray*10./21. + edges/21.).astype(np.uint8)
    # processed = (thresh*10/20. + gray*10./20.).astype(np.uint8)
    processed = (filtered/2. + gray/2.).astype(np.uint8)
    return processed
    # return gray
    # return edges

class ROIManager:
    def __init__(self, r1, c1, rsize, csize):
        self.r1 = r1
        self.r2 = r1 + rsize
        self.c1 = c1
        self.c2 = c1 + csize
        self.rsize = rsize
        self.csize = csize

    def get_cut_image(self, img):
        return img[self.c1:self.c2,self.r1:self.r2]
    
    def draw_roi(self, img):
        lw = 2
        cv2.rectangle(img, 
                      (self.r1-lw, self.c1-lw), 
                      (self.r2+lw, self.c2+lw), 
                      (255,0,0),
                      lw,
                      )

class MousePts:
    def __init__(self,windowname,img):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(windowname,img)
        self.curr_pt = []
        self.point   = []

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x,y])
            #print(self.point)
            cv2.circle(self.img,(x,y),1,(0,255,0),-1)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x,y]
            #print(self.point)

    def getpt(self,count=1,img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname,self.img)
        cv2.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        while(1):
            cv2.imshow(self.windowname,self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point)>=count:
                break
            #print(self.point)
        cv2.setMouseCallback(self.windowname, lambda *args : None)
        #cv2.destroyAllWindows()
        return self.point, self.img

if __name__=='__main__':
    img = np.zeros((512,512,3), np.uint8)
    windowname = 'image'
    coordinateStore = MousePts(windowname,img)

    pts,img = coordinateStore.getpt(3)
    print(pts)

    pts,img = coordinateStore.getpt(3,img)
    print(pts)

    cv2.imshow(windowname,img)
    cv2.waitKey(0)