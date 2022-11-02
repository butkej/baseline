import os
import csv
import numpy as np
import cv2

list_subtype = ["all"]
data_all = []
for subtype in list_subtype:
    tmp_id = np.loadtxt(
        f"./slideID_JMR/{subtype}.txt", delimiter=",", dtype="str"
    ).tolist()
    data_all = data_all + tmp_id

output = np.zeros((len(data_all), 3), dtype="object")

for i, slideID in enumerate(data_all):

    data_csv = np.genfromtxt(f"./csv_JMR/{slideID}.csv", delimiter=",", dtype="int")

    num_patch = data_csv.shape[0]

    fn_img = slideID + "_thumb.tif"
    thumb = cv2.imread(f"./thumb_JMR/{fn_img}")
    all_patch = int(thumb.shape[0] / 4 * thumb.shape[1] / 4)

    output[i, 0] = str(slideID)
    output[i, 1] = str(num_patch)
    output[i, 2] = str(all_patch)

fn_output = "./patch_JMR.csv"
np.savetxt(fn_output, output, delimiter=",", fmt="%s")
