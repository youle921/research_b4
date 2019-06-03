from PIL import Image
import glob
import os
import numpy as np

max_tnum = 144

files = sorted(glob.glob('./*.png'))
time = []
for path in files:
    time.append(os.path.getctime(path))
fix_files = np.array(files)[np.argsort(time)]

images = list(map(lambda file: Image.open(file), fix_files))

images[0].save('all_pic.gif', save_all=True, append_images=images[1:], duration=100,)
images[0].save('max_pic.gif', save_all=True, append_images=images[1:max_tnum], duration=100, )


