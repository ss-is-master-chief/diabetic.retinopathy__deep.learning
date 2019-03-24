import glob
import shutil
import os
import cv2
from tqdm import tqdm
from colorama import Fore, Style

import pandas as pd
import numpy as np

#from scripts.fetch_data import fetch_db0
from scripts.preprocess_2 import preprocess_image
from scripts.build_model import build_model
from scripts.get_classes import create_label

if __name__ == "__main__":
    #fetch_db0()

    dir_path = "./diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images"
    #dir_path=r"C:\Users\Ishan\Desktop\diaretdb0_v_1_1\resources\images\diaretdb0_fundus_images"
    dot_path= "./diaretdb0_v_1_1/resources/images/diaretdb0_groundtruths"
    #dot_path=r"C:\Users\Ishan\Desktop\diaretdb0_v_1_1\resources\images\diaretdb0_groundtruths"
    output_path = "./normalised_images"
    norm_images = pd.DataFrame()
    imagenorms = []

    if not os.path.exists('normalised_images'):
        os.makedirs('normalised_images')
    else:
        shutil.rmtree('normalised_images')
        os.makedirs('normalised_images')

    print("{}[~] Preprocessing Images into directory: normalised_images ..{}".format(Fore.YELLOW, Style.RESET_ALL))
    #print(os.listdir(dir_path))
    for file_name in tqdm(os.listdir(dir_path)):
        #print(file_name)
        if "png" in file_name:
            path = "{}\{}".format(dir_path, file_name)
            #print(path)
            output = "{}\{}".format(output_path, file_name)
            new_img = preprocess_image(path)
            cv2.imwrite(output, new_img)
            temp=cv2.imread(output)
            b,g,r=cv2.split(temp)
            #print(g.shape)
            imagenorms.append(np.reshape(g,(1,512*512)))

    print("\n{}[\u2713] Pre-processing complete..{}".format(Fore.GREEN, Style.RESET_ALL))

    labels = create_label(dot_path)
    norm_images = pd.DataFrame(np.squeeze(np.asarray(imagenorms)))
    norm_images['class'] = labels['class']

    x_train = norm_images.iloc[:,0:262144].values
    y_train = pd.get_dummies(norm_images['class']).values

    # x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.5)

    x_train = np.array([np.reshape(x, (512,512)) for x in x_train])
    # x_test = np.array([np.reshape(y, (512,512)) for y in x_test])

    x_train = x_train.reshape(x_train.shape[0], 512, 512, 1)
    # x_test = x_test.reshape(x_test.shape[0], 512, 512, 1)
    input_shape = (512, 512, 1)

    model, tensorboard = build_model()

    print("{}----- MODEL SUMMARY -----{}".format(Fore.GREEN, Style.RESET_ALL))
    print(model.summary())

    model.fit(x_train, y_train, 
                    epochs=100, 
                    batch_size=16, 
                    shuffle=True, 
                    validation_split=0.2,
                    callbacks=[tensorboard],
                    verbose=1)

    score_train = model.evaluate(x_train, y_train, verbose=1)
    print('{}Train loss: {}{}'.format(Fore.RED, score_train[0], Style.RESET_ALL))
    print('{}Train accuracy: {} %{}'.format(Fore.RED, score_train[1]*100, Style.RESET_ALL))

    

