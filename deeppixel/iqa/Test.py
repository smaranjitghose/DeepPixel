import brisque as b

path_hd = r"Images_Test\HD.jpg"
path_noisy= r"Images_Test\Noise.png"
path_blur= r"Images_Test\Blur.png"

def normal():
    img, gray = b.imgread(path_hd)
    b.imgscore(img,True)
    
def noisy():
    img, gray = b.imgread(path_noisy)
    b.imgscore(img,True)
    
def blur():
    img, gray = b.imgread(path_blur)
    b.imgscore(img,True)
    
normal()
noisy()
blur()
