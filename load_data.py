import os
import pickle
from PIL import Image
import numpy as np

def resize(image, desired_size, flag='image'):
    curr_size = image.size
    coef = float(desired_size)/max(curr_size)
    new_size = [int(x*coef) for x in curr_size]

    if flag == 'image':
        image = image.resize(new_size, Image.LANCZOS)
    else:
        image = image.resize(new_size, Image.NEAREST)
    
    new_image = Image.new(image.mode, (desired_size, desired_size))
    new_image.paste(image, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    return new_image

def load_images_and_masks(data_folder):
    images = {}
    masks = {}
    
    for folder_name in os.listdir(data_folder):
        institution_f = os.path.join(data_folder, folder_name)
        for curr_f in os.listdir(institution_f):
            image_path = os.path.join(institution_f, curr_f)
            if ('mask' not in image_path):
                mask_path = os.path.join(institution_f, image_path[:-4]+'_mask.tif')
                if os.path.isfile(image_path) and os.path.isfile(mask_path):
                    image = resize(Image.open(image_path).convert('RGB'), 1024)
                    mask = resize(Image.open(mask_path).convert('1'), 1024, 'mask')
                    images[folder_name] = np.array(image)
                    masks[folder_name] = np.array(mask)
    return images, masks

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

data_folder = 'D:\\MasterStudije\\2. Godina\ZimskiSemestar\\DDL\\Projekat\\mri_seg\\data\\'
size_of_images = (1024, 1024)
images, masks = load_images_and_masks(data_folder)
data = {'images': images, 'masks': masks}
print(len(data['images']))
print(len(data['masks']))
save_to_pickle(data, 'data.pkl')
