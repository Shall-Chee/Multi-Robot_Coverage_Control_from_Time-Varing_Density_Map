
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
from sklearn.cluster import KMeans
import argparse
# import random
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import time


class cvt():

    def __init__(self, distribution, robots_count, robot_positions):

        self.robots_count = robots_count
        self.distribution = distribution

        self.distribution = np.clip(self.distribution, 0.01, 255)

        # max_ = np.max(distribution)
        # max_ind = np.where(distribution >= max_ - 1) # np.max(a) = 254
        sorted_distribution = np.argsort(self.distribution.ravel())
        
        # print(sorted_distribution)
        sorted_index = np.dstack(np.unravel_index(sorted_distribution, (len(self.distribution),len(self.distribution[0]))))[0]
        max_ind = sorted_index[-robots_count:]

        # plt.clf()
        # v_plot = plt.imshow(a)
        # plt.gca().invert_yaxis()
        # colorbar = plt.colorbar()

        # plt.plot(max_ind[1], max_ind[0], 'ro')

        # robot_positions = np.random.random((robots_count,2)) * 255
        # robot_positions = np.zeros(5,2)

        # task_agents = self.robots_count


        self.robot_positions = robot_positions
        print(self.robot_positions)

        # self.robot_positions = np.array([[1, 40], [40, 50], [75, 0], [85, 0], [120, 115]])
#         self.robot_positions = np.array([[  6.31808567 ,  1.75807792],
#  [ 25.26284896 ,123.77443459],
#  [ 15.19486774  ,33.07753588],
#  [  1.24577142  ,60.85128164],
#  [ 70.66037511  ,61.82643254]])

        # self.robot_positions = np.array([[25,5],[30,30],[15,70],[37,110],[75,60]])
        # robot_positions = max_ind

        self.vor = Voronoi(np.array(self.robot_positions))
        fig = voronoi_plot_2d(self.vor)

        self.distribution = np.transpose(self.distribution)
        v_plot = plt.imshow(self.distribution)
        plt.gca().invert_yaxis()
        colorbar = plt.colorbar()

        # max_ind

        plt.plot(self.robot_positions.T[0], self.robot_positions.T[1], 'ro', label = 'robot_position')
        plt.xlim(0 - len(self.distribution) * 0.1, len(self.distribution) * 1.1)
        plt.ylim(0 - len(self.distribution) * 0.1, len(self.distribution) * 1.1)


    def voronoi_finite(self):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi Input diagram
        a: input arrary

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        ridge_points: ndarray of ints, shape (nridges, 2)
        Indices of the points between which each Voronoi ridge lies.

        ridge_vertices: list of list of ints, shape (nridges, *)
        Indices of the Voronoi vertices forming each Voronoi ridge.
        """
        
        new_regions = []
        new_vertices = self.vor.vertices.tolist()  #Coordinates of the Voronoi vertices.

        center = self.vor.points.mean(axis=0)

        # print(vor.ridge_points, vor.ridge_vertices)

        # Construct a map containing all ridges for a given point
        # Graph for each point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        # print(f"point_region: {vor.point_region}")
        for p1, region in enumerate(self.vor.point_region): #Index of the Voronoi region for each input point. 
            vertices = self.vor.regions[region] # Indices of the Voronoi vertices forming each Voronoi region


            # if all(v >= 0 for v in vertices):
            #     # finite region
            #     new_regions.append(vertices)
            #     continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            # calculate the far point
            for p2, v1, v2 in ridges:

                v1_coor = self.vor.vertices[v1]
                v2_coor = self.vor.vertices[v2]            

                # if v2 < 0 or self.outside_plot(v2_coor):
                if v2 < 0:
                    v1, v2 = v2, v1
                    v1_coor, v2_coor = v2_coor, v1_coor

                if v1 >= 0:
                    # finite ridge: already in the region
                    
                    # vertices outside the range
                    new_region, new_vertices = self.intersect_outside_vertices(v1_coor, v2_coor, new_region, new_vertices, v1)

                if v2 >= 0 and v1 >= 0:
                    new_region, new_vertices = self.intersect_outside_vertices(v2_coor, v1_coor, new_region, new_vertices, v2)

                if v1 < 0 and not self.outside_plot(v2_coor):

                    # Compute the missing endpoint of an infinite ridge

                    t = self.vor.points[p2] - self.vor.points[p1]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = self.vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n

                    x_direction = np.sign(direction[0])
                    y_direction = np.sign(direction[1])

                    xmax = len(self.distribution[0]) if x_direction > 0 else 0
                    ymax = len(self.distribution) if y_direction > 0 else 0

                    v2 = self.vor.vertices[v2]
                    x_intersect = (xmax - v2[0]) / direction[0] * direction[1] + v2[1]
                    y_intersect = (ymax - v2[1]) / direction[1] * direction[0] + v2[0]

                    if y_intersect <= 128 and y_intersect > 0 and (x_intersect > 128 or x_intersect < 0):
                        far_point = np.array([y_intersect, ymax])
                        plt.plot([y_intersect, v2[0]], [ymax, v2[1]], color = 'k')
                    else:
                        far_point = np.array([xmax, x_intersect])
                        plt.plot([xmax, v2[0]], [x_intersect, v2[1]], color = 'k')


                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                    # four corner points

            xmin = np.min(np.array(new_vertices)[new_region, 0])
            xmax = np.max(np.array(new_vertices)[new_region, 0])
            ymin = np.min(np.array(new_vertices)[new_region, 1])
            ymax = np.max(np.array(new_vertices)[new_region, 1])

            # new_vertices = np.array(new_vertices)
            # calculate the four corners
            if xmin <= 0 and ymin <= 0:
                new_region.append(len(new_vertices))
                new_vertices.append([0, 0])   
            elif xmin <= 0 and ymax >= len(self.distribution):
                new_region.append(len(new_vertices))
                new_vertices.append([0, len(self.distribution)])                          
            elif xmax >= len(self.distribution[0]) and ymin <= 0:
                new_region.append(len(new_vertices))
                new_vertices.append([len(self.distribution[0]), 0])     
            elif xmax >= len(self.distribution[0]) and ymax >= len(self.distribution):
                new_region.append(len(new_vertices))
                new_vertices.append([len(self.distribution[0]), len(self.distribution)])
            elif xmin <= 0 and xmax >= len(self.distribution[0]):
                # print(np.array(new_vertices)[new_region])
                new_vertices_coor = np.array(new_vertices)[new_region]
                new_vertices_sorted = new_vertices_coor[new_vertices_coor[:,0].argsort()]
                new_region.append(len(new_vertices))
                new_region.append(len(new_vertices) + 1)
                if np.arctan2((self.vor.points[p1] - new_vertices_sorted[0])[1], (self.vor.points[p1] - new_vertices_sorted[0])[0]) < \
                    np.arctan2((new_vertices_sorted[1] - new_vertices_sorted[0])[1] , (new_vertices_sorted[1] - new_vertices_sorted[0])[0]):
                    new_vertices.append([0, 0])
                    new_vertices.append([len(self.distribution[0]), 0])
                else:
                    new_vertices.append([0, len(self.distribution)])
                    new_vertices.append([len(self.distribution[0]), len(self.distribution)])                                     
            elif ymin <= 0 and ymax >= 128:
                # print(new_vertices[:,1].argsort())
                # print(np.array(new_vertices)[[new_region]][:,1].argsort())
                new_vertices_coor = np.array(new_vertices)[new_region]
                new_vertices_sorted = new_vertices_coor[new_vertices_coor[:,1].argsort()]
                new_region.append(len(new_vertices))
                new_region.append(len(new_vertices) + 1)

                # assume convex
                if np.arctan2((self.vor.points[p1] - new_vertices_sorted[0])[1], (self.vor.points[p1] - new_vertices_sorted[0])[0]) < \
                        np.arctan2((new_vertices_sorted[1] - new_vertices_sorted[0])[1], (new_vertices_sorted[1] - new_vertices_sorted[0])[0]):
                    new_vertices.append([len(self.distribution[0]), ymin])
                    new_vertices.append([len(self.distribution[0]), ymax])
                else:
                    new_vertices.append([0, 0])
                    new_vertices.append([0, len(self.distribution)])   



            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region]) # New indices of the Voronoi vertices forming each Voronoi region
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())
        return new_regions, new_vertices

    def intersect_outside_vertices(self, v1_coor, v2_coor, new_region, new_vertices, v1):
        if v1_coor[0] < 0 or v1_coor[0] > len(self.distribution[0]):
            xmax = len(self.distribution[0]) if v1_coor[0] > len(self.distribution[0]) else 0
            x_intersect = (xmax - v1_coor[0]) / (v2_coor[0] - v1_coor[0]) * (v2_coor[1] - v1_coor[1]) + v1_coor[1]
            if x_intersect >= 0 and x_intersect < len(distribution[0]):
                far_point = np.array([xmax, x_intersect])
                # print(v1_coor)
                # print(new_vertices)
                # ind = new_vertices.index(v1_coor)
                if v1 in new_region:
                    new_region.remove(v1)
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

        if v1_coor[1] < 0 or v1_coor[1] > len(self.distribution):
            ymax = len(self.distribution) if v1_coor[1] > len(self.distribution) else 0
            y_intersect = (ymax - v1_coor[1]) / (v2_coor[1] - v1_coor[1]) * (v2_coor[0] - v1_coor[0]) + v1_coor[0]
            if y_intersect >= 0 and y_intersect < len(distribution):
                far_point = np.array([y_intersect, ymax])
                # print(v1_coor)
                # print(new_vertices)
                # ind = new_vertices.index(v1_coor)
                if v1 in new_region:
                    new_region.remove(v1)
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

        return new_region, new_vertices

    def outside_plot(self, coord):
        if coord[0] < 0 or coord[0] > len(self.distribution[0]):
            return True
        if coord[1] < 0 or coord[1] > len(self.distribution):
            return True
        return False


    def compute_centroids(self, regions, vertices):

        """ Compute the Voronoi tesselation and the corresponding centroids """


        vertices = np.asarray(vertices)
        # print(regions)

        polygons = [vertices[t] for t in regions]

        # centroids = np.zeros((len(self.robot_positions), 2))
        centroids = []

        for i in range(len(polygons)):
        # for i in range(1):
            # extract grid extrema of the polygon
            polygon = polygons[i]
            xmin = np.min(polygon[:, 0])
            xmax = np.max(polygon[:, 0])
            ymin = np.min(polygon[:, 1])
            ymax = np.max(polygon[:, 1])
            
            # compute the centroid: equation 2 
            
            N = 0
            D = 0
            if len(polygon) <= 1:
                continue
            hull = Delaunay(polygon)

            for j in range(int(xmin), int(xmax)):
                for k in range(int(ymin), int(ymax)):
                    p = np.array([j, k])
                    # print(hull.find_simplex(p))
                    vo_ind = hull.find_simplex(p)
                    if vo_ind >= 0:
                        p = np.array([j, k])
                        if k < 0 or k > len(self.distribution) - 1:
                            continue
                        if j < 0 or j > len(self.distribution[0]) - 1:
                            continue
                        density = self.distribution[k, j]
                        if density == 0:
                            continue
                        N += np.multiply(p, density)
                        D += density
            # if D != 0:
            centroids.append(N / D)
        return np.array(centroids)

    def compute_h(self, regions, vertices, centroids):

        vertices = np.asarray(vertices)
        # plt.plot(vertices[:,0], vertices[:,1],'bo', label = 'vertices')
        # print(regions)

        polygons = [vertices[t] for t in regions]

        # centroids = np.zeros((len(self.robot_positions), 2))
        # centroids = []
        h = []
        # N = np.zeros(len(centroids))

        for i in range(len(polygons)):
        # for i in range(1):
            # extract grid extrema of the polygon
            polygon = polygons[i]
            xmin = np.min(polygon[:, 0])
            xmax = np.max(polygon[:, 0])
            ymin = np.min(polygon[:, 1])
            ymax = np.max(polygon[:, 1])
            
            # compute the centroid: equation 2 
            
            cost = 0
            # D = 0
            if len(polygon) <= 1:
                continue
            hull = Delaunay(polygon)

            for j in range(int(xmin), int(xmax)):
                for k in range(int(ymin), int(ymax)):
                    p = np.array([j, k])
                    # print(hull.find_simplex(p))
                    vo_ind = hull.find_simplex(p)
                    if vo_ind >= 0:
                        p = np.array([j, k])
                        if k < 0 or k > len(self.distribution) - 1:
                            continue
                        if j < 0 or j > len(self.distribution[0]) - 1:
                            continue
                        density = self.distribution[k, j]
                        if density == 0:
                            continue
                        cost += np.multiply((p - centroids[i])[0] ** 2 + (p - centroids[i])[1] ** 2, density)
                        # cost += density

            h.append(cost)
            #             D += density
            # if D != 0:
            #     centroids.append(N / D)
        return np.array(h)

robots_count = 6
distribution = np.load('/Users/xiaoqi/Downloads/Kumar Lab/lab material/target_distribution.npy')

x_min, x_max, y_min, y_max = 0, 127, 0, 127
bbx = np.asarray([x_min, x_max, y_min, y_max])
robot_positions = np.random.random((robots_count,2)) * (bbx[1::2] - bbx[0::2]) + bbx[0::2]

robot_positions = np.array(robot_positions)
# robot_positions = np.array([[6.31808567 ,  1.75807792],
# [ 25.26284896,123.77443459],
# [ 15.19486774,33.07753588],
# [  1.24577142,60.85128164],
# [ 70.66037511,61.82643254]])

# robot_positions = np.array([[  1.47235712 , 92.617959  ],
#  [ 82.7358943 ,  42.3577841 ],
#  [ 52.59757841 , 97.05911685],
#  [100.11363943 , 65.8821007 ],
#  [121.74576803  ,51.79748087]])
# ims = []
# fig2 = plt.figure()
for i in range(25):
    cvt_ = cvt(distribution, robots_count, robot_positions)

    regions, vertices = cvt_.voronoi_finite()
    vertices = np.asarray(vertices)
    plt.plot(vertices[:,0], vertices[:,1],'bo', label = 'vertices')
    centroids = cvt_.compute_centroids(regions, vertices)
    plt.plot(centroids.transpose()[0], centroids.transpose()[1], 'go', label = 'centroids')
    # print(centroids)
    # update
    diff = (robot_positions - centroids)
    diff_2 = diff * diff
    sum_ = np.sqrt(np.sum(diff_2, axis=1))
    new_robot_positions = centroids.copy()
    
    upper_bound = 10
    for i in range(len(sum_)):
        if sum_[i] > upper_bound:
            new_robot_positions[i] = robot_positions[i] + upper_bound / sum_[i] * (centroids[i] - robot_positions[i])


    # centroids = centroids.T


    # huristic
    cost = cvt_.compute_h(regions, vertices, centroids)
    min_ind = np.argmin(cost)
    max_ind = np.argmax(cost)
    diff_min_max = new_robot_positions[max_ind] - new_robot_positions[min_ind]
    diff_min_max_2 = sum(diff_min_max * diff_min_max)
    sum_local = np.sqrt(np.sum(diff_min_max_2))
    upper_bound_local = 35
    upper_bound_local_step = 20
    # if diff_min_max_2 > upper_bound:
    if sum_local > upper_bound_local:
        new_robot_positions[min_ind] =  new_robot_positions[min_ind] + upper_bound_local_step / sum_local * (new_robot_positions[max_ind] - new_robot_positions[min_ind])
    new_robot_positions = new_robot_positions.copy()
    plt.plot(new_robot_positions.transpose()[0], new_robot_positions.transpose()[1], 'co', label = 'new_robot_position')

    
    plt.quiver(robot_positions.transpose()[0], robot_positions.transpose()[1], (new_robot_positions - robot_positions).transpose()[0], (new_robot_positions - robot_positions).transpose()[1],angles='xy', scale_units='xy', scale=1)
    plt.legend(loc = "upper left")
    plt.title(f'number of robots = {robots_count}')
    plt.ion()
    plt.pause(0.001)  

    #break condition
    new_diff = robot_positions - new_robot_positions
    new_diff_2 = new_diff * new_diff
    sum_new = np.sum(new_diff_2)

    if sum_new < 0.1 or i == 24:
        plt.pause(5)
        break

    plt.close()
    robot_positions = new_robot_positions.copy()
    

    
    # ims.append((plt.plot(centroids[0], centroids[1], 'go', label = 'centroids')))


# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
#                                    blit=True)

# plt.show()

    