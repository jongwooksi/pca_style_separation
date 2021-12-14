import cv2
import numpy as np 
from PIL import Image

contentimg = cv2.imread('content1.jpg')
contentimg = cv2.resize(contentimg, (256,256))

styleimg = cv2.imread('style4.jpg')
styleimg = cv2.cvtColor(styleimg[30:,:], cv2.COLOR_BGR2RGB)
styleimg = cv2.resize(styleimg, (256,256))
styleimg = Image.fromarray(styleimg)
styleimg = np.array(styleimg)
styleimg_r, styleimg_g, styleimg_b = styleimg[:,:,0], styleimg[:,:,1], styleimg[:,:,2]


def comp_2d_style(img):
    Maxnumber = 0
    
    convolutionMatrix = img - np.mean(img , axis = 1)
    eigenValue, eigenVector = np.linalg.eigh(np.cov(convolutionMatrix))
    
    idx = np.argsort(eigenValue)
    idx = idx[::-1]

    
    eigenVector = eigenVector[:,idx]
    eigenValue = eigenValue[idx]
    

    for i in range(len(eigenValue)):
        rate = sum(eigenValue[:i]) / sum(eigenValue)
        
        if rate > 0.9:
            Maxnumber = i
            break
   
    print(Maxnumber, sum(eigenValue[0:Maxnumber]) / sum(eigenValue))

    if Maxnumber < np.size(eigenVector, axis =1) or Maxnumber >0:
        eigenVector = eigenVector[:, range(Maxnumber)]
        
    score = np.dot(eigenVector.T, convolutionMatrix)
    reconImg = np.dot(eigenVector, score) + np.mean(img, axis = 1).T 
    
    return np.uint8(np.absolute(reconImg))




pca_red, pca_green, pca_blue = comp_2d_style(styleimg_r), comp_2d_style(styleimg_g), comp_2d_style(styleimg_b) 

sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
contentimg = cv2.filter2D(contentimg, -1, sharpening_mask1)

styleimg=np.array(Image.fromarray(np.dstack((pca_red, pca_green, pca_blue)) )) 

styleimg=cv2.cvtColor(styleimg, cv2.COLOR_RGB2BGR)

styleimg = cv2.medianBlur(styleimg, 3)
styleimg = cv2.fastNlMeansDenoisingColored(styleimg,None,21,21,7,21)


#styleimg = cv2.imread('style.jpg')
#styleimg = cv2.cvtColor(styleimg, cv2.COLOR_BGR2RGB)
#styleimg = cv2.resize(styleimg, (256,256))
#averagefilter_mask = np.ones((7,7), np.float32)/49
#styleimg = cv2.filter2D(styleimg,-1,averagefilter_mask)
#styleimg = cv2.fastNlMeansDenoisingColored(styleimg,None,21,21,7,21)


dst = cv2.addWeighted(contentimg, 0.3, styleimg, 0.7, 0)
cv2.imshow('s', dst)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2YCrCb)
dst = cv2.split(dst)
dst[0] = cv2.equalizeHist(dst[0])
dst = cv2.merge(dst)
dst = cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)

cv2.imshow('result', dst)

key = cv2.waitKey(0)

cv2.destroyAllWindows()


