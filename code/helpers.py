import matplotlib.pyplot as plt
import os
import imageio
from scipy import ndimage
from scipy.spatial.distance import cdist

from param import *


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


def exp_map(distribution, alpha=5.0, beta=5e-4):
    """
    Description: add exponential term according to distance to the centroid of entire distribution distribution
    Input:
        distribution: density map
    Output:
        output: exponential density map
    """
    Yc, Xc = ndimage.measurements.center_of_mass(distribution)
    X, Y = np.meshgrid(np.arange(distribution.shape[1]), np.arange(distribution.shape[0]))

    exp_term = np.exp(-beta * ((X - Xc) ** 2 + (Y - Yc) ** 2))
    output = distribution + alpha * exp_term
    return output


def count_robot(distribution):
    """
    Description: compute a reasonable robot count according to the density map
    Input:
        distribution: distribution distribution
    Output:
        robot_cnt: number of robots
    """
    count_point = np.count_nonzero(distribution > distribution_lower_bound)
    robot_cnt = round(count_point / (distribution.sum() / area_per_robot))
    return robot_cnt


def collision_planner(robot_pos, alpha=1, beta=5e-4):
    """
    Description: generate repulsive force according to distances between robots
    Input:
        robot_pos: robot positions
    Output:
        output: repulsive force for each robot
    """
    N = len(robot_pos)

    dist = cdist(robot_pos, robot_pos)
    exp_dist = np.exp(-beta * dist)
    exp_dist[dist > 10] = 0

    ori = np.tile(robot_pos, (1, N)) - robot_pos.reshape(1, -1)  # robot_pos (N, 2)
    ori /= (np.repeat(dist, 2, axis=1) + eps)

    target = np.reshape(ori * np.repeat(exp_dist, 2, axis=1), (N, 2, -1)).sum(axis=-1)
    return alpha * target


def video_maker(video_name, png_dir, fig_index, start_flag, video_type="mp4", fpi=10, dpi=90, end_pause=1):
    """
    Description: generate video
    Input:
        video_name: name of video
        png_dir:    figure directory
        fig_index:  figure index
        start_flag: flag indication beginning of video-making process
        video_type: "mp4"/"gif"
        fpi:        fpi
        dpi:        dpi
        end_pause:  pause time of last frame
    """
    # make png path if it doesn't exist already
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # save each .png for GIF
    # lower dpi gives a smaller, grainier GIF; higher dpi gives larger, clearer GIF
    plt.savefig(os.path.join(png_dir, 'frame_' + str(fig_index)) + '_.png', dpi=dpi)
    # plt.close('all')  # comment this out if you're just updating the x,y data

    if start_flag:
        plt.close()
        print("Generating Video...")

        # sort the .png files based on index used above
        images, image_file_names = [], []
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png'):
                image_file_names.append(file_name)
        sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[1]))

        # define some GIF parameters
        if video_type == "mp4":
            writer = imageio.get_writer(video_name + ".mp4", fps=20)
        frame_length = 1.0 / fpi  # seconds between frames
        # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
        for ii in range(0, len(sorted_files)):
            file_path = os.path.join(png_dir, sorted_files[ii])
            if ii == len(sorted_files) - 1:
                for jj in range(0, int(end_pause / frame_length)):
                    if video_type == "gif": images.append(imageio.imread(file_path))
                    elif video_type == "mp4": writer.append_data(imageio.imread(file_path))
            else:
                if video_type == "gif": images.append(imageio.imread(file_path))
                elif video_type == "mp4": writer.append_data(imageio.imread(file_path))
        # the duration is the time spent on each image (1/duration is frame rate)
        if video_type == "gif": imageio.mimsave(video_name + ".gif", images, 'GIF', duration=frame_length)
        elif video_type == "mp4": writer.close()
        print("Finished Generating Video!")
