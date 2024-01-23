
class ImageOverlayBlending:
    def __init__(self, imgPath) -> None:
        self.imgPath:str = imgPath
        self.mix:float = 0
        cv2.namedWindow("OpenCV View") #
        cv2.createTrackbar('Mixing', 'OpenCV View', 0,100, lambda x:x) 
    
    def setMix(
        self,
        numpy_array
    ) -> np.array:
        numpy_array = cv2.resize(numpy_array, dsize=(1920, 1080), interpolation=cv2.INTER_AREA) #
        im2 = cv2.imread(self.imgPath) #
        im2 = cv2.resize(im2, dsize=(1920, 1080), interpolation=cv2.INTER_AREA) #
        img = cv2.addWeighted(numpy_array, float(100-self.mix)/100, im2 , float(self.mix)/100, 0) #
        self.mix = cv2.getTrackbarPos('Mixing','OpenCV View')
        return img