import numpy as np
from PIL import Image
import librosa as lib
import matplotlib.pyplot as plt


def main():
    path = '/Users/dmitriypetunin/Downloads/IMG_3142.JPG'
    music = '/Users/dmitriypetunin/Documents/songs/FACE_-_Labirint_(musmore.com).mp3'

    # img = np.asarray(Image.open(path).convert('RGB'))
    # print(img)

    y, sc = lib.load(music)
    plt.plot(range(len(y)),y)
    plt.show()




if __name__ == "__main__":
    main()