import numpy as np
from aicsimageio import AICSImage
from czifile import CziFile
from xml.etree import ElementTree as ET
import os
import re
import glob
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from tune_final import find_threshold
import tifffile

d_546_factor = 1.2
d_647_factor = 1.45

TARGET_DIR = "/home/laurids/Desktop/hcrs_coped/joe_thresholding"

def exp_func(x, a, b, c):
    """Expoentional with offset"""
    return a * np.exp(b * x) + c


def infer_naming_scheme(img_name):
    first_part = img_name.split(".czi")[0]
    first_part = first_part[:-2]
    last = first_part[-3:].lower()
    if "ch" in last:
        return "channel_last"
    else:
        return "channel_first"
    
def infer_channel_names(img_name, naming_scheme):
    img_name = img_name.lower()
    channel_dict = {0: None, 1: None, 2: None}
    if naming_scheme == "channel_last":
        img_name = img_name[::-1]
        remain1 = img_name.split("1hc")[1]
        first = remain1.find("_")
        second = remain1.find("_", first + 1)  
        channel1 = remain1[first+1:second]
        channel_dict[0] = channel1
        remain2 = img_name.split("2hc")[1]
        first = remain2.find("_")
        second = remain2.find("_", first + 1)  
        channel2 = remain2[first+1:second]
        channel_dict[1] = channel2
        remain3 = img_name.split("3hc")[1]
        first = remain3.find("_")
        second = remain3.find("_", first + 1) 
        channel3 = remain3[first+1:second]
        channel_dict[2] = channel3
    else:
        remain1 = img_name.split("ch1")[1]
        first = remain1.find("_")
        second = remain1.find("_", first + 1)  
        channel1 = remain1[first+1:second]
        channel_dict[0] = channel1
        remain2 = img_name.split("ch2")[1]
        first = remain2.find("_")
        second = remain2.find("_", first + 1)  
        channel2 = remain2[first+1:second]
        channel_dict[1] = channel2
        remain3 = img_name.split("ch3")[1]
        first = remain3.find("_")
        second = remain3.find("_", first + 1)  
        channel3 = remain3[first+1:second]
        channel_dict[2] = channel3
    return channel_dict


if __name__ == "__main__":
    all_files = os.listdir("/home/laurids/Desktop/hcrs_coped/examine")
    from tqdm import tqdm
    for agrp_filename in tqdm(all_files):
        img = AICSImage(os.path.join("/home/laurids/Desktop/hcrs_coped/examine", agrp_filename))
        img = img.data 
        with CziFile(os.path.join("/home/laurids/Desktop/hcrs_coped/examine", agrp_filename)) as czi:
            root = ET.fromstring(czi.metadata())
        channels = root.findall('.//Information/Image/Dimensions/Channels/Channel')
        channel_wave_lengths = {0: 0, 1: 1, 2: 2}
        for i, ch in enumerate(channels):
            name = ch.get('Name')
            dye  = ch.findtext('DyeName') or ch.findtext('Fluor')
            wave = int(dye.split(" ")[-1])
            channel_wave_lengths[i] = wave
            print(f"C={i}: id={ch.get('Id')} name={name} dye={dye}")
        img = img.squeeze()
        gcamp_channel = np.argmax(np.array([1 if value == 488 else 0 for key, value in channel_wave_lengths.items()]))
        channel_647 = np.argmax(np.array([1 if value == 647 else 0 for key, value in channel_wave_lengths.items()]))
        channel_546 = np.argmax(np.array([1 if value == 546 else 0 for key, value in channel_wave_lengths.items()]))
        gcamp_image = img[gcamp_channel]
        gcamp_image_dim_0_part = gcamp_image.shape[0] // 5
        gcamp_image_cropped = gcamp_image[gcamp_image_dim_0_part:-gcamp_image_dim_0_part]
        y_fit = np.percentile(gcamp_image_cropped, (75), axis=(1, 2))
        x_fit = np.arange(len(y_fit))
        #Inital guess
        p0_guess = [1.0, 0.01, 5]
        #bounds for opt
        lower_bounds = [0, 0, 00]
        upper_bounds = [np.inf, 5, 100]

        popt_robust, pcov_robust = curve_fit(
            exp_func, x_fit, y_fit,
            p0=p0_guess,
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            maxfev=10000,
            bounds=(lower_bounds, upper_bounds)
        )
        best_a = popt_robust[0]
        best_b = popt_robust[1]
        best_c = popt_robust[2]
        stretched_vals_546 = exp_func(np.arange(len(gcamp_image)), best_a, best_b * d_546_factor, best_c)
        stretched_vals_647 = exp_func(np.arange(len(gcamp_image)), best_a, best_b * d_647_factor, best_c)
        naming_scheme = infer_naming_scheme(agrp_filename)
        channel_name_dict = infer_channel_names(agrp_filename, naming_scheme)

        img_546 = img[channel_546]
        img_647 = img[channel_647]

        thresholded_546 = find_threshold(img_546, stretched_vals_546)
        thresholded_647 = find_threshold(img_647, stretched_vals_647)

        full_name = agrp_filename.split(".czi")[0]


        saving_546_name = channel_name_dict[channel_546] + full_name + "thresholded.tif"
        saving_647_name = channel_name_dict[channel_647] + full_name + "thresholded.tif"
        tifffile.imwrite(os.path.join(TARGET_DIR, saving_546_name), thresholded_546)
        tifffile.imwrite(os.path.join(TARGET_DIR, saving_647_name), thresholded_647)


