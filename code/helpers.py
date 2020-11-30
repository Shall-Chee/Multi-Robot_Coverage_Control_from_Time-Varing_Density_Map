import numpy as np
import matplotlib.pyplot as plt
import os, imageio


def interp(start, end, ratio):
    """
    Description: interpolation between start point set and end point set
    Input:
        start: start point set, (N, 2) or (2,)
        end: end point set, (N, 2) or (2,)
        ratio: ratio, (N, 2) or (N,)
    Output:
        mid_point: interpolated point set, (N, 2)
    Note: if ratio is (N,) vector or start/end is single point (2,), use broadcast
    """
    if isinstance(ratio, np.ndarray):
        N = len(ratio)  # ratio is array
    else:
        N = 1  # ratio is scalar
    start = np.array(start).reshape(-1, 2)
    end = np.array(end).reshape(-1, 2)
    ratio = np.array(ratio).reshape(N, -1)

    assert len(start) in [1, N]
    assert len(end) in [1, N]
    assert ratio.shape[1] in [1, 2]

    return start + (end - start) * ratio


def euler_dist(pt1, pt2):
    """
    Description: distance between two point sets
    Input:
        pt1: point set 1, (N, 2) or (2, N)
        pt2: point set 1, (N, 2) or (2, N)
    Output:
        dist: distance, (N, 2)
    """
    pt1 = np.array(pt1).reshape(-1, 2)
    pt2 = np.array(pt2).reshape(-1, 2)
    dist = np.sqrt(np.sum((pt2 - pt1) ** 2), axis=1)

    return dist


def gif_maker(gif_name, png_dir, gif_index, start_flag, fpi=10, dpi=90, end_pause=1):
    # make png path if it doesn't exist already
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # save each .png for GIF
    # lower dpi gives a smaller, grainier GIF; higher dpi gives larger, clearer GIF
    plt.savefig(png_dir + 'frame_' + str(gif_index) + '_.png', dpi=dpi)
    # plt.close('all')  # comment this out if you're just updating the x,y data

    if start_flag:
        plt.close()
        print("Making GIF...")

        # sort the .png files based on index used above
        images, image_file_names = [], []
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png'):
                image_file_names.append(file_name)
        sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[1]))

        # define some GIF parameters

        frame_length = 1.0 / fpi  # seconds between frames
        # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
        for ii in range(0, len(sorted_files)):
            file_path = os.path.join(png_dir, sorted_files[ii])
            if ii == len(sorted_files) - 1:
                for jj in range(0, int(end_pause / frame_length)):
                    images.append(imageio.imread(file_path))
            else:
                images.append(imageio.imread(file_path))
        # the duration is the time spent on each image (1/duration is frame rate)
        imageio.mimsave(gif_name, images, 'GIF', duration=frame_length)
        print("Finished Making GIF!")
