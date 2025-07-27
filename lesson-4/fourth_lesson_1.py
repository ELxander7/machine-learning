import librosa
import matplotlib.pyplot as plt

def main():
    y, sr = librosa.load("D:\sample-3s.wav")
    plt.plot(range(len(y),sr))
    plt.show()
    librosa
    print(sr)

if __name__ == '__main__':
    main()