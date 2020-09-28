from yolo import YOLO
from PIL import Image

yolo = YOLO()

# predict image
while True:

    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)

        # save predict image as predict_img.png
        r_image.save("predict_img.png","png")
        r_image.show()