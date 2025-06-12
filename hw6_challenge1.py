from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Union, Tuple, List
import os
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import keyboard

def generateIndexMap(gray_list: List[np.ndarray], w_size: int) -> np.ndarray:
    # Generate an index map for the refocusing application
    # Input:
    #   gray_list - List of K gray-scale images
    #   w_size - half window size used for focus measure computation
    # Output:
    #   index_map - mxn index map
    #               index_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)

    def find_best_focused_layer(focus_measure_stack, window_size):
        # Smooth the focus measure data using a moving average filter
        smoothed_focus_measure_stack = uniform_filter(focus_measure_stack, size=(1,window_size,window_size))
        
        # Find the layer with the maximum focus measure for each pixel
        index_map = np.argmax(smoothed_focus_measure_stack, axis=0)
        
        return index_map

    def compute_modified_laplacian(image):
        # Define the modified Laplacian kernel
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        
        # Compute the modified Laplacian response using convolution
        modified_laplacian = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')
        
        return modified_laplacian
    
    def compute_focus_measure(image, window_size):
        # Compute the modified Laplacian response
        modified_laplacian = compute_modified_laplacian(image)
        
        # Square the modified Laplacian response
        modified_laplacian_squared = np.square(modified_laplacian)
        
        # Compute the sum of squared responses within a small window for each pixel
        focus_measure = np.zeros_like(image)
        half_window = window_size // 2
        for i in range(half_window, image.shape[0] - half_window):
            for j in range(half_window, image.shape[1] - half_window):
                window = modified_laplacian_squared[i - half_window:i + half_window + 1,
                                                     j - half_window:j + half_window + 1]
                focus_measure[i, j] = np.sum(window)
        
        return focus_measure

    focus_measure_stack = []
    for gray_img in gray_list:
        focus_measure_map = compute_focus_measure(gray_img, w_size * 2)
        # Image.fromarray((focus_measure_map * 255).astype(np.uint8)).save('pqr/index_map.png')
        focus_measure_stack.append(focus_measure_map)
    
    index_map = find_best_focused_layer(focus_measure_stack, 14)
    return index_map
    # raise NotImplementedError


def loadFocalStack(focal_stack_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Load the focal stack
    # Input:
    #   focal_stack_dir - directory of the focal stack
    # Output:
    #   rgb_list - List of RGB images for varying focal lengths
    #   gray_list - List of gray-scale images for varying focal lengths

    files = os.listdir(focal_stack_dir)
    rgb_list = []; gray_list = []
    for i in range(1, len(files) + 1):
      # Construct the full file path
      file_path = os.path.join(focal_stack_dir, f'frame{i}.jpg')
      
      # Load the image
      rgb_image = Image.open(file_path)
      
      # Convert the RGB image to grayscale
      grayscale_image = rgb_image.convert("L")

      rgb_image = np.array(rgb_image) / 255.0
      grayscale_image = np.array(grayscale_image) / 255.0
      rgb_list.append(rgb_image)
      gray_list.append(grayscale_image)
    return rgb_list, gray_list
    # raise NotImplementedError


def refocusApp(rgb_list: List[np.ndarray], depth_map: np.ndarray) -> None:
    # Refocusing application
    # Input:
    #   rgb_list - List of RGB images for varying focal lengths
    #   depth_map - mxn index map
    #               depth_map(i, j) is the index of the image that is in focus
    #               at pixel (i, j)

    focal_stack = rgb_list
    index_map = depth_map
    num_layers = len(focal_stack)
    height = focal_stack[0].shape[0]
    width = focal_stack[0].shape[1]
    
    layer_index = 0
    plt.figure()
    while True:
        # Display an image in the focal stack
        print(layer_index)

        plt.imshow(focal_stack[layer_index])
        # plt.title("Layer {}".format(layer_index))
        plt.xlabel("Choose a scene point (click on image)")
        scene_point = plt.ginput(n=1, timeout=-1, show_clicks=True)
        if not scene_point:
            print("No scene point selected. Exiting...")
            break
        
        scene_point = np.round(scene_point[0]).astype(int)
        if not (0 <= scene_point[1] < height and 0 <= scene_point[0] < width):
            print("Selected point is outside the image frame. Exiting...")
            break
        # print(scene_point[1], scene_point[0])
        # print(round(index_map[scene_point[1], scene_point[0]]))
        layer_index = round(index_map[scene_point[1], scene_point[0]])
        plt.figure(clear = True)
        if keyboard.is_pressed('esc'):
            break



    # raise NotImplementedError
