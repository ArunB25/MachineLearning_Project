from hashlib import new
from PIL import Image
import os


def resize_image(final_size, im):
    '''
    Resizes the image and uses the RGB channels for colour
    '''
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':

    if not os.path.exists('formated_images'):
        os.makedirs('formated_images')
    path = "images/"
    dirs = os.listdir(path)
    dirs.sort()
    final_size = 512
    for item in dirs[:1]:
        im = Image.open('images/' + item)
        new_im = resize_image(final_size, im)
        item_name = item.replace('.', '')
        new_im.save(f'formated_images/{item_name}_resized.jpg')
        new_im.show()
        print(f"image: {item} Formatted")
