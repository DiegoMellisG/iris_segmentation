import datagenerator
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from utils import class_mean_iou

iris = [255,128,0]
pupil = [0,255,0]
sclera = [0,255,255]
bg = [0,0,0]

colours = [bg, iris, pupil, sclera]
colour_codes = np.array(colours)

def reverse_one_hot(one_hot_matrix, colour_codes):
    argmax = np.argmax(one_hot_matrix, axis=-1)
    img = colour_codes[argmax.astype(int)]

    return img

def generate_dataframe(dataset, predictions):
    image_id = 0
    class_ious = []
    for img_batches, mask_batches in dataset:
        for i in range(len(img_batches)):
            class_m_iou = class_mean_iou(mask_batch[i], predictions[image_id])
            class_m_iou['filename'] = dataset.image_info[image_id]['id'][:-4]
            pred_img = colour_codes[np.argmax(pred[image_id], axis = -1).astype(int)]
            gt_img = colour_codes[np.argmax(mask_batch[i], axis = -1).astype(int)]
            class_m_iou['pred_img'] = pred_img
            class_m_iou['gt_img'] = gt_img
            class_ious.append(class_m_iou)
            image_id+=1
    df = pd.DataFrame(class_ious)
    return df

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
    test_dataset.load_eyes('60_20_20/dataset','test')
    test_dataset.prepare()
    print("Image Count (Test): {}".format(len(test_dataset.image_ids)))

    pred = trained_model.predict(test_dataset)
    iou_df = generate_dataframe(test_dataset, pred)

    if not os.path.isdir("CSV"):
        os.makedirs("CSV"))
    
    iou_df.to_csv('CSV/IoU.csv')

    
    
if __name__ == '__main__':
    main()
