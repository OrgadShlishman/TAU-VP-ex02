import os
import cv2
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from lucas_kanade import lucas_kanade_optical_flow, warp_image, \
    lucas_kanade_step


# FILL IN YOUR ID
ID1 = '123456789'
ID2 = '987654321'

# Choose parameters
WINDOW_SIZE_RIVER = 5  # Add your value here!
MAX_ITER_RIVER = 5  # Add your value here!
NUM_LEVELS_RIVER = 3

# Output dir and statistics file preparations:
RIVER_DIR = 'river_results'
os.makedirs(RIVER_DIR, exist_ok=True)
STATISTICS_PATH = f'RIVER_{ID1}_{ID2}_mse_and_time_stats.json'


statistics = OrderedDict()


def calc_mse_at_interest_region(
        first_image: np.ndarray, second_image: np.ndarray, interest_size: int
        = WINDOW_SIZE_RIVER//2) -> float:
    """Calculate the Mean Squared Error (Difference) between two images in
    the interest region.

    Args:
        first_image: np.ndarray. First image.
        second_image: np.ndarray. Second image.
        interest_size: int. The number of rows and cols to cut from top,
        bottom, left and right.

    Returns:
        mse: float. The Mean Squared Error (Difference) between the two
        images in the interest region.
    """
    first_image_interesting_part = first_image[
                                   interest_size:-interest_size,
                                   interest_size:-interest_size]
    second_image_interesting_part = second_image[
                                   interest_size:-interest_size,
                                   interest_size:-interest_size]
    squared_difference = (first_image_interesting_part -
                          second_image_interesting_part)**2
    mse = squared_difference.mean()
    return mse


# Load images I1,I2
I1 = cv2.cvtColor(cv2.imread('river1.png'), cv2.COLOR_RGB2GRAY)
I2 = cv2.cvtColor(cv2.imread('river2.png'), cv2.COLOR_RGB2GRAY)

# Compute optical flow using LK algorithm
start_time = time.time()
(du, dv) = lucas_kanade_step(I1, I2, WINDOW_SIZE_RIVER)
end_time = time.time()
statistics['[RIVER, TIME] One Step LK'] = end_time - start_time

# Warp I2
I2_one_lk_step = warp_image(I2, du, dv)

# The MSE should decrease as the warped image (I2_warp) should be similar to I1
original_mse = calc_mse_at_interest_region(I1, I2, WINDOW_SIZE_RIVER // 2)
after_one_lk_step_mse = calc_mse_at_interest_region(I1,
                                                    I2_one_lk_step,
                                                    WINDOW_SIZE_RIVER // 2)

print(f'MSE of original frames: {original_mse}')
print(f'MSE after one LK step: {after_one_lk_step_mse}')
print(f'MSE ratio one step LK: {original_mse / after_one_lk_step_mse}')
print(f'One LK-step took: {end_time - start_time:.2f}[sec]')
statistics['[RIVER, MSE] Original video'] = original_mse
statistics['[RIVER, MSE] One Step LK'] = after_one_lk_step_mse


one_step_warped_image = warp_image(I2, du, dv)
plt.subplot(2, 2, 1)
plt.title('du')
plt.imshow(du, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('dv')
plt.imshow(dv, cmap='gray')
plt.subplot(2, 3, 4)
plt.title('I1')
plt.imshow(I1, cmap='gray')
plt.subplot(2, 3, 5)
plt.title('I2 warped to I1')
plt.imshow(one_step_warped_image, cmap='gray')
plt.subplot(2, 3, 6)
plt.title('I2')
plt.imshow(I2, cmap='gray')
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.suptitle('One LK step')
plt.savefig(os.path.join(RIVER_DIR, '0_river_one_LK_step_result.png'))

# create river gifs:
cv2.imwrite(os.path.join(RIVER_DIR, 'river1.png'), I1.astype(np.uint8))
cv2.imwrite(os.path.join(RIVER_DIR, 'river2.png'), I2.astype(np.uint8))
cv2.imwrite(os.path.join(RIVER_DIR, 'river2_warped.png'),
            one_step_warped_image.astype(np.uint8))
image_paths = [os.path.join(RIVER_DIR, x)
               for x in ['river1.png', 'river2.png']]
images = (Image.open(f) for f in image_paths)
img = next(images)
img.save(fp=os.path.join(RIVER_DIR, '1_original.gif'), format='GIF',
         append_images=images, save_all=True, duration=200, loop=0)
image_paths = [os.path.join(RIVER_DIR, x)
          for x in ['river1.png', 'river2_warped.png']]
images = (Image.open(f) for f in image_paths)
img = next(images)
img.save(fp=os.path.join(RIVER_DIR, '2_after_one_lk_step.gif'),
         format='GIF', append_images=images, save_all=True, duration=200,
         loop=0)

################################################################################
######################### ONE STEP LUCAS KANADE ENDS HERE ######################
################################################################################

# calculate LK optical flow:
start_time = time.time()
(u, v) = lucas_kanade_optical_flow(I1, I2, WINDOW_SIZE_RIVER, MAX_ITER_RIVER,
                                   NUM_LEVELS_RIVER)
end_time = time.time()
statistics['[RIVER, TIME] Full LK'] = end_time - start_time

I2_full_lk = warp_image(I2, u, v)
after_full_lk_mse = calc_mse_at_interest_region(I1,
                                                I2_full_lk,
                                                WINDOW_SIZE_RIVER // 2)
print(f'MSE of original frames: {original_mse}')
print(f'MSE after full LK: {after_full_lk_mse}')
print(f'MSE ratio full LK: {original_mse / after_full_lk_mse}')
print(f'Full LK-step took: {end_time - start_time:.2f}[sec]')
statistics['[RIVER, MSE] Full LK'] = after_full_lk_mse


plt.subplot(2, 2, 1)
plt.title('du')
plt.imshow(du, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('dv')
plt.imshow(dv, cmap='gray')
plt.subplot(2, 3, 4)
plt.title('I1')
plt.imshow(I1, cmap='gray')
plt.subplot(2, 3, 5)
plt.title('I2 warped to I1')
plt.imshow(I2_full_lk, cmap='gray')
plt.subplot(2, 3, 6)
plt.title('I2')
plt.imshow(I2, cmap='gray')
fig = plt.gcf()
fig.set_size_inches(8, 8)
plt.suptitle('Full Lucas-Kanade Algorithm')
plt.savefig(os.path.join(RIVER_DIR, 'river_full_LK_step_result.png'))

# greate full LK algorithm result for river image as a gif:
cv2.imwrite(os.path.join(RIVER_DIR, 'river2_warped_full_lk.png'),
            I2_full_lk.astype(np.uint8))
image_paths = [os.path.join(RIVER_DIR, x)
          for x in ['river1.png', 'river2_warped_full_lk.png']]
images = (Image.open(f) for f in image_paths)
img = next(images)
img.save(fp=os.path.join('river_results', '3_after_full_lk.gif'),
         format='GIF', append_images=images, save_all=True, duration=200,
         loop=0)

with open(STATISTICS_PATH, 'w') as f:
    json.dump(statistics, f, indent=4)
