import datagenerator
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
import numpy as np
import os

iris = [255,128,0]
pupil = [0,255,0]
sclera = [0,255,255]
bg = [0,0,0]

colours = [bg, iris, pupil, sclera]
colour_codes = np.array(colours)

def reverse_one_hot(one_hot_matrix, colour_codes):
    pred_argmax = np.argmax(one_hot_matrix, axis=-1)
    pred_img = colour_codes[pred_argmax.astype(int)]

    return pred_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trained_model_filename', type=str, help='Trained model filename')
    args = parser.parse_args()
    trained_model_filename = os.path.abspath(args.trained_model_filename)

    # Check inputs
    if not os.path.exists(trained_model_filename):
        print('Not found:', trained_model_filename)
        sys.exit(-1)
    
    # Load the model
    trained_model = load_model(trained_model_filename)

    # Load Test Dataset
    test_dataset = datagenerator.EyeDataset(batch_size=10, dim=(120, 160), shuffle = False)
    test_dataset.load_eyes('dataset','test')
    test_dataset.prepare()
    print("Image Count (Test): {}".format(len(test_dataset.image_ids)))

    pred = trained_model.predict(test_dataset)
    

if __name__ == '__main__':
    main()
