import cv2
import time
import json
from collections import OrderedDict
from lucas_kanade import lucas_kanade_video_stabilization, \
    lucas_kanade_faster_video_stabilization, \
    lucas_kanade_faster_video_stabilization_fix_effects, get_video_parameters


# FILL IN YOUR ID
ID1 = '123456789'
ID2 = '987654321'

# Choose parameters
WINDOW_SIZE_TAU = 5  # Add your value here!
MAX_ITER_TAU = 5  # Add your value here!
NUM_LEVELS_TAU = 5  # Add your value here!


# Output dir and statistics file preparations:
STATISTICS_PATH = f'TAU_VIDEO_{ID1}_{ID2}_mse_and_time_stats.json'
statistics = OrderedDict()


def calc_mean_mse_video(path: str) -> float:
    """Calculate the mean MSE across all frames.

    The mean MSE is computed between every two consecutive frames in the video.

    Args:
        path: str. Path to the video.

    Returns:
        mean_mse: float. The mean MSE.
    """
    input_cap = cv2.VideoCapture(path)
    video_info = get_video_parameters(input_cap)
    frame_amount = video_info['frame_count']
    input_cap.grab()
    # extract first frame
    prev_frame = input_cap.retrieve()[1]
    # convert to greyscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mse = 0.0
    for i in range(1, frame_amount):
        input_cap.grab()
        frame = input_cap.retrieve()[1]  # grab next frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mse += ((frame - prev_frame) ** 2).mean()
        prev_frame = frame
    mean_mse = mse / (frame_amount - 1)
    return mean_mse


# Load video file
input_video_name = 'input.avi'

output_video_name = f'{ID1}_{ID2}_stabilized_video.avi'
start_time = time.time()
lucas_kanade_video_stabilization(input_video_name,
                                 output_video_name,
                                 WINDOW_SIZE_TAU,
                                 MAX_ITER_TAU,
                                 NUM_LEVELS_TAU)
end_time = time.time()
print(f'LK-Video Stabilization Taking all pixels into account took: '
      f'{end_time - start_time:.2f}[sec]')
statistics["[TAU, TIME] naive LK implementation"] = end_time - start_time

faster_output_video_name = f'{ID1}_{ID2}_faster_stabilized_video.avi'
start_time = time.time()
lucas_kanade_faster_video_stabilization(input_video_name,
                                        faster_output_video_name,
                                        WINDOW_SIZE_TAU,
                                        MAX_ITER_TAU,
                                        NUM_LEVELS_TAU)
end_time = time.time()
print(f'LK-Video Stabilization FASTER implementation took: '
      f'{end_time - start_time:.2f}[sec]')
statistics["[TAU, TIME] FASTER LK implementation"] = end_time - start_time

fixed_image_borders_output_video_name = f'{ID1}_{ID2}_' \
                                        f'fixed_borders_stabilized_video.avi'
start_time = time.time()
lucas_kanade_faster_video_stabilization_fix_effects(
    input_video_name, fixed_image_borders_output_video_name, WINDOW_SIZE_TAU,
    MAX_ITER_TAU, NUM_LEVELS_TAU, start_rows=10, start_cols=2, end_rows=30, end_cols=30)
end_time = time.time()
print(f'LK-Video Stabilization FASTER implementation took WITHOUT BORDERS: '
      f'{end_time - start_time:.2f}[sec]')
statistics["[TAU, TIME] FASTER, WITHOUT BORDERS LK implementation "] = \
    end_time - start_time



print("The Following MSE values should make sense to you:")
original_mse = calc_mean_mse_video(input_video_name)
print(f"Mean MSE between frames for original video: {original_mse:.2f}")
naive_mse = calc_mean_mse_video(output_video_name)
print(f"Mean MSE between frames for Lucas Kanade Stabilized output video: "
      f"{naive_mse:.2f}")
faster_mse = calc_mean_mse_video(faster_output_video_name)
print(f"Mean MSE between frames for Lucas Kanade Stabilized output FASTER "
      f"Implementation video: {faster_mse:.2f}")
faster_no_borders_mse = calc_mean_mse_video(
    fixed_image_borders_output_video_name)
print(f"Mean MSE between frames for Lucas Kanade Stabilized output FASTER "
      f"Implementation + BORDERS CUT video: {faster_no_borders_mse:.2f}")

statistics["[TAU, MSE] original video "] = original_mse
statistics["[TAU, MSE] naive implementation "] = naive_mse
statistics["[TAU, MSE] FASTER implementation "] = faster_mse
statistics["[TAU, MSE] FASTER, WITHOUT BORDERS LK implementation "] = \
    faster_no_borders_mse

with open(STATISTICS_PATH, 'w') as f:
    json.dump(statistics, f, indent=4)
