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
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

def imagej_auto_contrast(image, saturated=0.35): #TODO make perc explicit
    image = image.astype(np.float32)
    flat = image.flatten()
    n_pixels = len(flat)
    saturated_pixel_count = int(n_pixels * saturated / 100.0)
    saturated_pixel_count = min(saturated_pixel_count, n_pixels // 2 - 1)  # avoid index overflow
    sorted_pixels = np.sort(flat)
    # Handle very small saturation count
    if saturated_pixel_count == 0:
        min_val = sorted_pixels[0]
        max_val = sorted_pixels[-1]
    else:
        min_val = sorted_pixels[saturated_pixel_count]
        max_val = sorted_pixels[-saturated_pixel_count - 1]
    # Prevent division by zero
    if max_val == min_val:
        return np.clip(image, 0, 1)
        # Stretch contrast
    stretched = (image - min_val) / (max_val - min_val)
    stretched = np.clip(stretched, 0, 1)

    return stretched

d_546_factor = 1.2
d_647_factor = 1.45

TARGET_DIR = "/home/laurids/Desktop/hcrs_coped/joe_thresholding"

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c



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
    fraction_brightness_corr_applied = []
    for agrp_filename in tqdm(all_files):
        # if "6dpf_huc_h2b_gcamp6s_ch3_nr3c1_ch2_nr3c2_ch1_GC6s_2"  in agrp_filename: # not
        #     continue
        img = AICSImage(os.path.join("/home/laurids/Desktop/hcrs_coped/examine", agrp_filename))
        img = img.data 
        with CziFile(os.path.join("/home/laurids/Desktop/hcrs_coped/examine", agrp_filename)) as czi:
            root = ET.fromstring(czi.metadata())
            # text = czi.metadata() if isinstance(czi.metadata(), str) else ET.tostring(czi.metadata(), encoding="unicode")
            # text = text.lower()

            # root
        channels = root.findall('.//Information/Image/Dimensions/Channels/Channel')
        channel_wave_lengths = {0: 0, 1: 1, 2: 2}
        
        flag = root.find(".//ZStackSetup/StackBrightnessCorrection")
        brightness_correction = (flag.text or "").strip().lower() == "true"
        corr_bool = 1 if brightness_correction else 0

        #import pdb; pdb.set_trace()

        fraction_brightness_corr_applied.append(corr_bool)


       # print("stackbrightnesscorrections" in text)   # the actual test, no slicing
        # text[text.index("stackbrightnesscorrections"): text.index("stackbrightnesscorrections") + 500]
        #print(root[i:i+60])
        #import pdb; pdb.set_trace()
        # print("node", node)
        # continue
        # StackBrightnessCorrection
        # Experiment|AcquisitionBlock|ZStackSetup|StackBrightnessCorrection = false
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
        img_488 = gcamp_image


        if not brightness_correction:
            # apply some rough brightness correction for the imagejautocontrast stuff
            gcamp_image_dim_0_part = gcamp_image.shape[0] // 5
            gcamp_image_cropped = gcamp_image[gcamp_image_dim_0_part:-gcamp_image_dim_0_part]
            y_fit = np.percentile(gcamp_image_cropped, (90), axis=(1, 2))
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

            stretched_vals_488 = exp_func(np.arange(len(gcamp_image)), best_a, best_b, best_c) - best_c
            img_488_adjusted = img_488 / stretched_vals_488[:, None, None]

        else:
            img_488_adjusted = img_488

        # one_two_vals = np.arange(1, 2, img_488.shape[0])

        # img_488_adjusted = img_488 / one_two_vals[:, None, None]

        for _ in range(6):
            img_488_adjusted = imagej_auto_contrast(img_488_adjusted)


        th = threshold_otsu(img_488_adjusted)
        res = gaussian_filter((img_488_adjusted > th).astype(np.float32), sigma = 10)
        th2 = threshold_otsu(res)
        contrast_threshold_result = res > th2


        # --- path length: tissue voxels ABOVE each voxel (light enters at z=0 top) ---
        column_sum_until_z = np.cumsum(contrast_threshold_result[::-1], axis=0)[::-1]
        column_sum_until_z[contrast_threshold_result == 0] = 0

        tissue = contrast_threshold_result.astype(bool)
        path_vals = column_sum_until_z[tissue]      # 1D: path length per tissue voxel
        inten_vals = img_488[tissue]                # 1D: matching 488 intensity

        # --- equal-count bins over tissue path lengths (n+1 edges for n bins) ---
        edges = np.unique(np.percentile(path_vals, np.linspace(0, 100, 101)))
        bin_idx = np.digitize(path_vals, edges[1:-1])

        x_fit, y_fit = [], []
        MIN_POP = 50                                 # drop sparse deep bins
        for k in range(len(edges) - 1):
            m = bin_idx == k
            if m.sum() < MIN_POP:
                continue
            x_fit.append(np.median(path_vals[m]))           # bin center, path units
            y_fit.append(np.median(inten_vals[m]))  # your 75th, per bin
        x_fit = np.asarray(x_fit, float)
        y_fit = np.asarray(y_fit, float)

        # --- robust exponential fit (same logic as the per-z version) ---
        p0_guess = [1.0, 0.01, 5]
        lower_bounds = [0, 0, 0]
        upper_bounds = [np.inf, 5, 100]
        popt_robust, pcov_robust = curve_fit(
            exp_func, x_fit, y_fit,
            p0=p0_guess, method='trf', loss='soft_l1', f_scale=1.0,
            maxfev=10000, bounds=(lower_bounds, upper_bounds),
        )
        best_a, best_b, best_c = popt_robust

        img_546 = img[channel_546]
        img_647 = img[channel_647]

        img_546[contrast_threshold_result == 0] = 0
        img_647[contrast_threshold_result == 0] = 0

        naming_scheme = infer_naming_scheme(agrp_filename)
        channel_name_dict = infer_channel_names(agrp_filename, naming_scheme)

        # IMPORTANT: only the multiplicative part a*exp(-b*L) attenuates light.
        # c is an additive floor (background/offset) and must NOT be divided out.
        atten = exp_func(column_sum_until_z, best_a, best_b, best_c) - best_c
        atten /= atten.max()                         # normalize: shallow tissue ~ 1
        atten = np.clip(atten, 1e-3, None)           # guard against blow-up at depth
        img_488_corrected = np.where(tissue, img_488 / atten, img_488)

        # --- HCR channel: SAME path field, wavelength-scaled decay rate ---
        atten_546 = exp_func(column_sum_until_z, best_a, best_b * d_546_factor, best_c) - best_c
        atten_546 /= atten_546.max()
        atten_546 = np.clip(atten_546, 1e-3, None)
        img_546_corrected = np.where(tissue, img_546 / atten_546, img_546)

        # --- HCR channel: SAME path field, wavelength-scaled decay rate ---
        atten_647 = exp_func(column_sum_until_z, best_a, best_b * d_647_factor, best_c) - best_c
        atten_647 /= atten_647.max()
        atten_647 = np.clip(atten_647, 1e-3, None)
        img_647_corrected = np.where(tissue, img_647 / atten_647, img_647)

        # gcamp_image_dim_0_part = gcamp_image.shape[0] // 5
        # gcamp_image_cropped = gcamp_image[gcamp_image_dim_0_part:-gcamp_image_dim_0_part]
        # y_fit = np.percentile(gcamp_image_cropped, (75), axis=(1, 2))
        # x_fit = np.arange(len(y_fit))
        # #Inital guess
        # p0_guess = [1.0, 0.01, 5]
        # #bounds for opt
        # lower_bounds = [0, 0, 00]
        # upper_bounds = [np.inf, 5, 100]

        # popt_robust, pcov_robust = curve_fit(
        #     exp_func, x_fit, y_fit,
        #     p0=p0_guess,
        #     method='trf',
        #     loss='soft_l1',
        #     f_scale=1.0,
        #     maxfev=10000,
        #     bounds=(lower_bounds, upper_bounds)
        # )
        # best_a = popt_robust[0]
        # best_b = popt_robust[1]
        # best_c = popt_robust[2]
        # stretched_vals_546 = exp_func(np.arange(len(gcamp_image)), best_a, best_b * d_546_factor, best_c)
        # stretched_vals_647 = exp_func(np.arange(len(gcamp_image)), best_a, best_b * d_647_factor, best_c)
        # stretched_vals_488 = exp_func(np.arange(len(gcamp_image)), best_a, best_b, best_c)
        # naming_scheme = infer_naming_scheme(agrp_filename)
        # channel_name_dict = infer_channel_names(agrp_filename, naming_scheme)

        # img_546 = img[channel_546]
        # img_647 = img[channel_647]
        # img_488 = gcamp_image

        # img_546 = img_546 / stretched_vals_546[:, None, None]
        # img_647 = img_647 / stretched_vals_647[:, None, None]
        # img_488 = img_488 / stretched_vals_488[:, None, None]

        # sampling_points = [[325, 1232, 414], [255, 746, 332], [185, 482, 581], [99, 558, 85], [23, 598, 48]]

        # sampled_vals = []
        # for point in sampling_points:
        #     vals = img_488[point[0]-12:point[0]+12, point[1]-12:point[1]+12, point[2]-12:point[2]+12]
        #     median_val = np.median(vals)
        #     sampled_vals.append(median_val)
        # sampled_vals = np.array(sampled_vals)
        full_name = agrp_filename.split(".czi")[0]

        saving_488_name = channel_name_dict[gcamp_channel] + full_name + "z_normed.tif"
        tifffile.imwrite(os.path.join(TARGET_DIR, saving_488_name), img_488_corrected)

        thresholded_546 = find_threshold(img_546_corrected)
        thresholded_647 = find_threshold(img_647_corrected)

        saving_546_name = channel_name_dict[channel_546] + full_name + "thresholded.tif"
        saving_647_name = channel_name_dict[channel_647] + full_name + "thresholded.tif"
        tifffile.imwrite(os.path.join(TARGET_DIR, saving_546_name), thresholded_546)
        tifffile.imwrite(os.path.join(TARGET_DIR, saving_647_name), thresholded_647)

    fraction_brightness_corr_applied = np.array(fraction_brightness_corr_applied)
    print("fraction brightness correction for 25 images", fraction_brightness_corr_applied.sum() / len(fraction_brightness_corr_applied))


