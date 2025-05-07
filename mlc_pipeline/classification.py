# mlc_pipeline/classification.py
import cv2
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import os
import matplotlib.pyplot as plt
import logging

class Classifier:
    def __init__(self, config):
        self.num_samples = config.get("num_samples", 5000)

    def classify(self, merged_image, output_dir=None):
        # Convert the image to a NumPy array if it isn't already.
        merged_image = np.array(merged_image)
    
        logging.info("Performing Maximum Likelihood Classification (MLC)...")
        red_channel = merged_image[:, :, 0]
        blue_channel = merged_image[:, :, 2]
        water_candidate = (blue_channel > 225) & (blue_channel != red_channel)
        nonwater_candidate = ~water_candidate
    
        water_indices = np.argwhere(water_candidate)
        nonwater_indices = np.argwhere(nonwater_candidate)
        num_water = min(self.num_samples, water_indices.shape[0])
        num_nonwater = min(self.num_samples, nonwater_indices.shape[0])
        np.random.shuffle(water_indices)
        np.random.shuffle(nonwater_indices)
        water_indices = water_indices[:num_water]
        nonwater_indices = nonwater_indices[:num_nonwater]
    
        X, y = [], []
        for i, j in water_indices:
            X.append([red_channel[i, j], blue_channel[i, j]])
            y.append(1)
        for i, j in nonwater_indices:
            X.append([red_channel[i, j], blue_channel[i, j]])
            y.append(0)
        X, y = np.array(X), np.array(y)
    
        clf = QuadraticDiscriminantAnalysis(reg_param=0.1)
        clf.fit(X, y)
    
        H_img, W_img, _ = merged_image.shape
        flat_red = merged_image[:, :, 0].reshape(-1, 1)
        flat_blue = merged_image[:, :, 2].reshape(-1, 1)
        X_all = np.hstack((flat_red, flat_blue))
        predictions = clf.predict(X_all).reshape(H_img, W_img)
        mlc_mask = (predictions.astype(np.uint8)) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(mlc_mask, kernel, iterations=8)
        morphed_mask = cv2.erode(dilated_mask, kernel, iterations=8)
        morphed_mask = cv2.dilate(morphed_mask, kernel, iterations=1)
    
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mlc_result_path = os.path.join(output_dir, "mlc_result.png")
            plt.imsave(mlc_result_path, morphed_mask, cmap="coolwarm")
            logging.info(f"Saved MLC result image to {mlc_result_path}")
    
        # Extract main water regions using connectivity and skeletonization.
        water_bool = morphed_mask > 0
        labeled_water = label(water_bool, connectivity=2)
        main_river_mask = np.zeros_like(morphed_mask)
        tol = 10
        H_mask, W_mask = morphed_mask.shape
        for region in regionprops(labeled_water):
            minr, minc, maxr, maxc = region.bbox
            region_mask = (labeled_water[minr:maxr, minc:maxc] == region.label)
            region_skel = skeletonize(region_mask)
            skel_coords = np.argwhere(region_skel)
            skel_coords[:, 0] += minr
            skel_coords[:, 1] += minc
            if (np.any(skel_coords[:, 0] <= tol) or np.any(skel_coords[:, 0] >= H_mask - tol) or
                np.any(skel_coords[:, 1] <= tol) or np.any(skel_coords[:, 1] >= W_mask - tol)):
                main_river_mask[labeled_water == region.label] = 255
    
        main_river_mask = cv2.dilate(main_river_mask, kernel, iterations=2)
        if output_dir:
            final_mask_path = os.path.join(output_dir, "final_mask.png")
            plt.imsave(final_mask_path, main_river_mask, cmap="Blues")
            logging.info(f"Saved final water mask image to {final_mask_path}")
        return main_river_mask

