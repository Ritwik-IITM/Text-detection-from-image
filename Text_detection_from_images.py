import cv2
import matplotlib.pyplot as plt
import easyocr

# Image reading
img_path="D:\Projects\Text_detection_with_python_OpenCV\OIP (5).jpg"
img=cv2.imread(img_path)

# Instance text detector
text_reader=easyocr.Reader(['en'])

# Text detection from image
txt=text_reader.readtext(img)
threshold=0.75
# drawing bbox around the text
for t in txt:
    bbox,text,score=t

    if score>threshold:
        cv2.rectangle(img,bbox[0],bbox[2],(0,0,255),5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
