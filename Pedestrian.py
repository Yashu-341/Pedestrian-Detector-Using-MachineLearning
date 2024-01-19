import cv2
import imutils
import tkinter as tk
from tkinter import filedialog

# Initializing the HOG person
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open a file dialog and get the image file path
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename()  # Open the file dialog

# Reading the Image
image = cv2.imread(file_path)

# image=fname
print(type(image))

# Resizing the Image
image = imutils.resize(image, width=min(800, image.shape[1]))

# Detecting all the regions in the Image that has a pedestrians inside it
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

# Drawing the regions in the Image
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Output Image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
