from PIL import Image
import numpy as np

img_f1 = r"D:\kidden\mt\open\ocr-syn-data\test\eng_word_data\images\000000071.jpg"

img1 = Image.open(img_f1)
print(img1.size)
# img1.show()

img_a = np.asarray(img1)
print(img_a.shape)
print(img_a)

img_pad = np.ones(shape=(32, 280)) * 255
img_pad[:, :img_a.shape[1]] = img_a

img_new = Image.fromarray(img_pad)
img_new.show()

# img1 = img1.resize((280, 32), Image.ANTIALIAS)
# print(img1.size)
# img1.show()
#
#
# img1.save("new2.jpg")
# img1 = Image.open("new2.jpg")
# print(img1.size)
# img1.show()
