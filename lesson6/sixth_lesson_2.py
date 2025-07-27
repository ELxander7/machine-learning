# import pandas as pd
#
# def main():
#     disease = pd.read_csv('disease.csv', sep = ";")
#     symptom = pd.read_csv('symptom.csv', sep = ";")
#
#     print(symptom.shape)
#
#     patient = [1,3,5]
#     #probabilities = [1.] * len(disease.shape[0])
#
#
#
# if __name__ == '__main__':
#     main()
import pandas as pd
import numpy as np


def main():
    disease = pd.read_csv('disease.csv', sep = ";")
    symptom = pd.read_csv('symptom.csv', sep = ";")

    # print(symptom.shape)

    patient = [1,3,5]
    probabilities = [1.] * (disease.shape[0]-1)
    for i in range(disease.shape[0]-1):
        probabilities[i] *= disease.loc[i][1]
        for j in range(symptom.shape[0]):
            if j in patient:
                probabilities[i] *= symptom.loc[j][i+1]
    print(disease.loc[np.argmax(probabilities)][0])

if __name__ == '__main__':
    main()

