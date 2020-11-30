import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay

from param import *
from helpers import *


class CVT:
    def __init__(self, distribution, robot_cnt, robot_pos):
        self.robot_cnt = robot_cnt
        self.distribution = np.clip(distribution.T, DENSITY_MIN, DENSITY_MAX)
        self.vor = Voronoi(robot_pos)
        self.regions, self.vertices = self.voronoi_finite(self.vor)
        self.centroids = self.compute_centroids(self.regions, self.vertices)

    def voronoi_finite(self, vor):
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
        new_vertices = vor.vertices.tolist()  # Coordinates of the Voronoi vertices.

        center = vor.points.mean(axis=0)
        # Construct a map containing all ridges for a given point
        # Graph for each point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        self.new_lines = []

        # Reconstruct infinite regions
        # print(f"point_region: {vor.point_region}")
        for p1, region in enumerate(vor.point_region):  # Index of the Voronoi region for each input point.
            vertices = vor.regions[region]  # Indices of the Voronoi vertices forming each Voronoi region

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            # calculate the far point
            for p2, v1, v2 in ridges:

                v1_coor = vor.vertices[v1]
                v2_coor = vor.vertices[v2]

                if v2 < 0:
                    v1, v2 = v2, v1
                    v1_coor, v2_coor = v2_coor, v1_coor

                if v1 >= 0:
                    # finite ridge: already in the region

                    # vertices outside the range
                    new_region, new_vertices = self.intersect_outside_vertices(v1_coor, v2_coor, new_region,
                                                                               new_vertices, v1)

                if v2 >= 0 and v1 >= 0:
                    new_region, new_vertices = self.intersect_outside_vertices(v2_coor, v1_coor, new_region,
                                                                               new_vertices, v2)

                if v1 < 0 and not self.outside_plot(v2_coor):

                    # Compute the missing endpoint of an infinite ridge

                    t = vor.points[p2] - vor.points[p1]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n

                    x_direction = np.sign(direction[0])
                    y_direction = np.sign(direction[1])

                    xmax = len(self.distribution[0]) if x_direction > 0 else 0
                    ymax = len(self.distribution) if y_direction > 0 else 0

                    v2 = vor.vertices[v2]
                    x_intersect = (xmax - v2[0]) / direction[0] * direction[1] + v2[1]
                    y_intersect = (ymax - v2[1]) / direction[1] * direction[0] + v2[0]

                    if y_intersect <= Y_MAX and y_intersect > Y_MIN and (x_intersect > X_MAX or x_intersect < X_MIN):
                        far_point = np.array([y_intersect, ymax])
                        self.new_lines.append(([y_intersect, v2[0]], [ymax, v2[1]]))
                        # plt.plot([y_intersect, v2[0]], [ymax, v2[1]], color='k')
                    else:
                        far_point = np.array([xmax, x_intersect])
                        self.new_lines.append(([xmax, v2[0]], [x_intersect, v2[1]]))
                        # plt.plot([xmax, v2[0]], [x_intersect, v2[1]], color='k')

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
                new_vertices_coor = np.array(new_vertices)[new_region]
                new_vertices_sorted = new_vertices_coor[new_vertices_coor[:, 0].argsort()]
                new_region.append(len(new_vertices))
                new_region.append(len(new_vertices) + 1)
                if np.arctan2((vor.points[p1] - new_vertices_sorted[0])[1],
                              (vor.points[p1] - new_vertices_sorted[0])[0]) < \
                        np.arctan2((new_vertices_sorted[1] - new_vertices_sorted[0])[1],
                                   (new_vertices_sorted[1] - new_vertices_sorted[0])[0]):
                    new_vertices.append([0, 0])
                    new_vertices.append([len(self.distribution[0]), 0])
                else:
                    new_vertices.append([0, len(self.distribution)])
                    new_vertices.append([len(self.distribution[0]), len(self.distribution)])
            elif ymin <= 0 and ymax >= 128:
                new_vertices_coor = np.array(new_vertices)[new_region]
                new_vertices_sorted = new_vertices_coor[new_vertices_coor[:, 1].argsort()]
                new_region.append(len(new_vertices))
                new_region.append(len(new_vertices) + 1)

                # assume convex
                if np.arctan2((vor.points[p1] - new_vertices_sorted[0])[1],
                              (vor.points[p1] - new_vertices_sorted[0])[0]) < \
                        np.arctan2((new_vertices_sorted[1] - new_vertices_sorted[0])[1],
                                   (new_vertices_sorted[1] - new_vertices_sorted[0])[0]):
                    new_vertices.append([len(self.distribution[0]), ymin])
                    new_vertices.append([len(self.distribution[0]), ymax])
                else:
                    new_vertices.append([0, 0])
                    new_vertices.append([0, len(self.distribution)])

                    # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in
                             new_region])  # New indices of the Voronoi vertices forming each Voronoi region
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())
        return new_regions, np.array(new_vertices)

    def intersect_outside_vertices(self, v1_coor, v2_coor, new_region, new_vertices, v1):
        if v1_coor[0] < 0 or v1_coor[0] > len(self.distribution[0]):
            xmax = len(self.distribution[0]) if v1_coor[0] > len(self.distribution[0]) else 0
            x_intersect = (xmax - v1_coor[0]) / (v2_coor[0] - v1_coor[0]) * (v2_coor[1] - v1_coor[1]) + v1_coor[1]
            if x_intersect >= 0 and x_intersect < len(self.distribution):
                far_point = np.array([xmax, x_intersect])
                if v1 in new_region:
                    new_region.remove(v1)
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

        if v1_coor[1] < 0 or v1_coor[1] > len(self.distribution):
            ymax = len(self.distribution) if v1_coor[1] > len(self.distribution) else 0
            y_intersect = (ymax - v1_coor[1]) / (v2_coor[1] - v1_coor[1]) * (v2_coor[0] - v1_coor[0]) + v1_coor[0]
            if y_intersect >= 0 and y_intersect < len(self.distribution):
                far_point = np.array([y_intersect, ymax])
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
        """ 
        Compute the Voronoi tesselation and the corresponding centroids 
        """
        vertices = np.asarray(vertices)
        polygons = [vertices[t] for t in regions]
        centroids = []

        for polygon in polygons:
            if len(polygon) <= 1:
                continue

            # extract grid extrema of the polygon
            xmin = int(np.min(polygon[:, 0]))
            xmax = int(np.max(polygon[:, 0]))
            ymin = int(np.min(polygon[:, 1]))
            ymax = int(np.max(polygon[:, 1]))

            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
            points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

            # compute the centroid: equation 2
            hull = Delaunay(polygon)
            indices = hull.find_simplex(points)
            density = np.where(indices >= 0, self.distribution[points[:, 1], points[:, 0]], 0)
            N = (points * density.reshape(-1, 1)).sum(axis=0)
            D = density.sum()
            centroids.append(N / D)
        return np.array(centroids)

    def compute_h(self, regions, vertices, centroids):
        vertices = np.asarray(vertices)
        polygons = [vertices[t] for t in regions]
        h = []

        for i in range(len(polygons)):
            polygon = polygons[i]
            if len(polygon) <= 1:
                continue

            # extract grid extrema of the polygon
            polygon = polygons[i]
            xmin = int(np.min(polygon[:, 0]))
            xmax = int(np.max(polygon[:, 0]))
            ymin = int(np.min(polygon[:, 1]))
            ymax = int(np.max(polygon[:, 1]))

            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
            points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

            # compute the centroid: equation 2
            hull = Delaunay(polygon)
            indices = hull.find_simplex(points)
            density = np.where(indices >= 0, self.distribution[points[:, 1], points[:, 0]], 0)
            cost = (((points - centroids[i])**2).sum(axis=1) * density).sum()
            h.append(cost)
        return np.array(h)

    def step(self, state):
        self.vor = Voronoi(state)
        self.regions, self.vertices = self.voronoi_finite(self.vor)
        self.centroids = self.compute_centroids(self.regions, self.vertices)


if __name__ == "__main__":
    # Read Distribution/Density Map
    distribution = np.load('./target_distribution.npy')
    scatter_ratio = np.random.random((robot_cnt, 2))
    robot_pos = interp([X_MIN, Y_MIN], [X_MAX, Y_MAX], scatter_ratio)  # generate robot on random initial positions

    # Simulate
    global_flag = True  # flag indicating whether finishing allocation or not
    done = False  # flag indicating whether reach terminate state
    timestep = 0
    fig = plt.figure()
    ax = plt.gca()
    x0, y0 = 0, 0

    # Divide Regions on Distribution Map
    cvt = CVT(distribution, robot_cnt, robot_pos)

    while not done:
        # Local Planner: Towards Centroid
        local_error = np.sqrt(np.sum((cvt.centroids - robot_pos) ** 2, axis=1))
        local_planner = interp(robot_pos, cvt.centroids, (local_move_limit / (local_error + eps)).clip(max=1)) - robot_pos

        # Global Planner: Better Allocate Cost
        cost = cvt.compute_h(cvt.regions, cvt.vertices, cvt.centroids)
        cost_order = np.argsort(cost)
        min_ind, max_ind = cost_order[0], cost_order[-1]
        if cost[min_ind] / cost[max_ind] > cost_ratio_thresh: global_flag = False  # finish reallocation
        global_error = np.sqrt(np.sum((robot_pos[max_ind] - robot_pos[min_ind]) ** 2))
        global_planner = np.zeros_like(robot_pos)
        global_planner[min_ind] = interp(robot_pos[min_ind], robot_pos[max_ind],
                                         (global_move_limit / (global_error + eps)).clip(max=1)) - robot_pos[min_ind]

        # Get Total Action and Step
        action = local_scale * local_planner + global_flag * global_scale * global_planner
        robot_pos += action
        timestep += 1
        cvt.step(robot_pos)
        done = np.linalg.norm(action) < pos_error_thresh or timestep > max_timestep
        print("Timestep: {}  Error: {:.4f}".format(timestep, np.linalg.norm(action)))

        # Render
        # voronoi_plot_2d(Voronoi(np.vstack((robot_pos, DUMMY_CORNER))), ax=ax)
        voronoi_plot_2d(cvt.vor, ax=ax)
        for line in cvt.new_lines:
            plt.plot(line[0], line[1], 'k')
        v_plot = plt.imshow(cvt.distribution)
        plt.gca().invert_yaxis()
        plt.plot((robot_pos - action)[:, 0], (robot_pos - action)[:, 1], 'ro', label='robot_position')
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.xlim(X_MIN - X_SIZE * 0.1, X_MAX + X_SIZE * 0.1)
        plt.ylim(Y_MIN - Y_SIZE * 0.1, Y_MAX + Y_SIZE * 0.1)
        plt.plot(cvt.vertices[:, 0], cvt.vertices[:, 1], 'bo', label='vertices')
        plt.plot(cvt.centroids[:, 0], cvt.centroids[:, 1], 'go', label='centroids')
        plt.plot(robot_pos[:, 0], robot_pos[:, 1], 'co', label='new_robot_position')
        plt.quiver(robot_pos[:, 0], robot_pos[:, 1], action[:, 0], action[:, 1], angles='xy', scale_units='xy', scale=1)
        plt.legend(loc="upper left")
        plt.title(f'number of robots = {robot_cnt}')
        plt.pause(0.1)
        plt.cla()

        # Terminal State
        if done: break

    plt.close()
