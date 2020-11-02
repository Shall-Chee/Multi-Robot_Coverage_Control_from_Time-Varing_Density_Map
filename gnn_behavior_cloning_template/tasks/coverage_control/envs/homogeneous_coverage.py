from enum import Enum
import gym
from gym import spaces
import numpy as np
from scipy.spatial import Voronoi, Delaunay
import matplotlib.pyplot as plt
import networkx as nx


class DensityType(Enum):
    GAUSSIAN = 1
    RING = 2
    # MULTI_GAUSSIAN = 3


class HomogeneousCoverageEnv(gym.Env):

    def __init__(self):
        ############### Agent Initialization #########################
        # number states per agent
        self.nx_system = 2
        # number of observations per agent
        self.n_features = 6
        # number of actions per agent
        self.nu = 2

        # default problem parameters
        self.n_agents = 10

        # robot position
        self.x = None
        self.u = None

        # Voronoi Information:
        self.centroids = None
        self.polygons = None

        # adjacency matrix
        self.A = np.zeros((self.n_agents, self.n_agents))
        self.proximity_graph = None
        ############### Time and Space initialization ######################
        self.region = np.array([[-5, -5, 5, 5], [-5, 5, 5, -5]])  # (clockwise starting from bottom left corner)
        self.bounds = np.array(
            [self.region[0, 0], self.region[0, 2], self.region[1, 0], self.region[1, 1]])  # xmin, xmax, ymin, ymax
        self.exploded_bounds = np.multiply(self.bounds, 5)  # for voronoi related purposes
        # max velocity
        self.vel_max = 500.0
        # step size
        self.dt = 0.033

        ####### density related initializations ###################
        # establish the method to use for densities
        self.density_type = DensityType.GAUSSIAN

        # Ring distribution (FIX)
        if self.density_type == DensityType.RING:
            self.ring_center = np.array([0, 0])
            self.ring_radius = 3.0

        # Gaussian distribution (FIX)
        if self.density_type == DensityType.GAUSSIAN:
            self.gaussian_mu = np.array([0, 3])
            self.gaussian_sigma = [1, 1]

        # grid points used to evaluate the density function
        self.num_grid_points = 50
        self.grid_finess = (self.bounds[1] - self.bounds[0]) / self.num_grid_points
        self.nx = np.linspace(self.bounds[0], self.bounds[1], self.num_grid_points)
        self.ny = np.linspace(self.bounds[2], self.bounds[3], self.num_grid_points)
        self.gridx, self.gridy = np.meshgrid(self.nx, self.ny, indexing='ij')
        self.grid_points = np.array([self.gridx.ravel(), self.gridy.ravel()]).T

        self.set_action_obs_space()

        self.fig = None

        self.seed()

    def set_action_obs_space(self):
        self.action_space = spaces.Box(low=-self.vel_max, high=self.vel_max, shape=(self.nu * self.n_agents,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

    def params_from_cfg(self, args):
        """
        Overwrite existing parameters with args
        Parameters
        ----------
        args: the .cfg

        Returns
        -------
        None
        """

        self.n_agents = args.getint('n_agents')
        self.set_action_obs_space()
        self.dt = args.getfloat('dt')

    def reset(self):
        """ Reset system to default state """
        self.x = np.zeros((self.n_agents, self.nx_system))

        # initialize the robots randomly
        self.x[:, 0] = np.random.uniform(self.bounds[0], self.bounds[1], self.n_agents)
        self.x[:, 1] = np.random.uniform(self.bounds[2], self.bounds[3], self.n_agents)
        self.centroids = self.compute_centroids()
        self.compute_proximity_graph()
        return self.get_obs()

    def compute_centroids(self):
        """ Compute the Voronoi tesselation and the corresponding centroids """
        # compute the voronoi cells based on robot positions
        vor = Voronoi(self.x)
        # create a finite-space version of the voronoi tesellation
        tessel, vertices = voronoi_finite_polygons_2d(vor,
                                                      self.exploded_bounds[1])

        self.polygons = [vertices[t] for t in tessel]
        # can be a lot more efficient
        centroids = np.zeros((self.n_agents, self.nx_system))
        for ii in range(self.n_agents):
            # extract grid extrema of the polygon
            xmin, xmax, ymin, ymax = extract_extrema(self.polygons[ii])
            xmin_ind = np.searchsorted(self.nx, xmin)
            xmax_ind = np.searchsorted(self.nx, xmax)
            ymin_ind = np.searchsorted(self.ny, ymin)
            ymax_ind = np.searchsorted(self.ny, ymax)
            # compute the centroid: equation 2 in https://ieeexplore.ieee.org/abstract/document/8901076
            N = 0
            D = 0
            hull = Delaunay(self.polygons[ii])
            # TODO: make list comprehension for speed
            for jj in range(xmin_ind, xmax_ind):
                for kk in range(ymin_ind, ymax_ind):
                    p = np.array([self.gridx[jj, kk], self.gridy[jj, kk]])
                    if hull.find_simplex(p) >= 0:
                        PHI = self.get_density_value(p)
                        N = N + np.multiply(p, PHI)
                        D = D + PHI
            centroids[ii, :] = N / (D)
        return centroids

    def is_common_vertex(self, ii, jj):
        """ Checks if the common vertices between the Voronoi cells of robot ii and jj lie within the domain """

        common_vertices = np.array(
            [x for x in set(tuple(x) for x in self.polygons[ii]) & set(tuple(x) for x in self.polygons[jj])])
        val = False
        for jj in range(common_vertices.shape[0]):
            if self.bounds[0] <= common_vertices[jj, 0] <= self.bounds[1] and self.bounds[2] <= common_vertices[
                jj, 1] <= self.bounds[3]:
                val = True
                break
        return val

    def compute_proximity_graph(self):
        """ Compute the proximity graph based on the Delaunay triangulation
            TODO(Sid): finish implementation
        """

        # compute the DeLaunay triangulation
        triangulation = Delaunay(self.x)
        # get the neighbors
        helper = triangulation.vertex_neighbor_vertices
        index_pointers = helper[0]
        indices = helper[1]
        # construct the adjacency matrix
        self.A = np.zeros((self.n_agents, self.n_agents))
        for ii in range(self.n_agents):
            self.A[ii, ii] = 1  # adding a self loop (do we need this?)
            neigh_list = indices[index_pointers[ii]:index_pointers[ii + 1]]
            for jj in neigh_list:
                if self.is_common_vertex(ii, jj):
                    self.A[ii, jj] = 1

        # create the networkx graph
        self.proximity_graph = nx.from_numpy_matrix(self.A)

    def get_obs(self):
        """
        Get the observation of the current state.
        Returns
        -------
        Tuple (rel_centroid, neighbors)
            rel_centroids: (n_agents x dim array) the relative distance vector from the robot to the weighted centroid
            rel_neighbors: (n_agents x (m neighbors x dim) list)The relative distance vector from the robot to all nearby neighbors
            TODO: make actual neighbors
        """

        self.centroids = self.compute_centroids()
        rel_centroids = self.centroids - self.x

        # Compute the proximity graph
        self.compute_proximity_graph()

        # TODO: optimize since neighbors is an overlapping calculation. Should also just spit out the relative values.
        rel_neighbors = [get_rel_neighbors(x_i, self.x, np.Inf) for x_i in self.x]

        return rel_centroids, rel_neighbors, self.proximity_graph

    def get_density_value(self, p):
        if self.density_type == DensityType.GAUSSIAN:
            return density_function_gaussian(self.gaussian_mu, self.gaussian_sigma, p)
        elif self.density_type == DensityType.RING:
            return density_function_ring(self.ring_center, self.ring_radius, p)
        # elif self.density_type == DensityType.MULTI_GAUSSIAN:
        #     return self.density_function_multiple_gaussian(p)
        else:
            raise Exception('Unknown DensityType %s' % str(self.density_type))

    def dynamics(self, action):
        # Simple Integration
        self.x = self.x + self.dt * action

    def step(self, action):
        """ Step through one timestep
        Parameters
        ------
        action: (n_agents,) array of actions for each agent.
        """
        clipped_action = np.clip(action, -self.vel_max, self.vel_max)
        self.dynamics(clipped_action)
        info = None
        done = False  # TODO
        reward = self.get_reward()
        return self.get_obs(), reward, done, info

    def get_reward(self):
        """ Returns the reward for the entire ensemble of robots by calculating the coverage integral"""
        # TODO: unimplemented
        print("Unimplemented reward calculation")
        return 0.0

    def close(self):
        pass

    def render(self, kwargs):

        if self.fig == None:
            plt.ion()  ## Note this correction
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            # self.ax.set_aspect('equal')
            self.ax.set_xlim(self.bounds[0], self.bounds[1])
            self.ax.set_ylim(self.bounds[2], self.bounds[3])
            self.robot_handle = self.ax.scatter(self.x[:, 0], self.x[:, 1], 20, 'black')
            self.centroid_handle = self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], 20, marker='s',
                                                   color='green')

            self.polygon_handle = []
            for ii in range(len(self.polygons)):
                self.polygon_handle.append(
                    self.ax.fill(*zip(*self.polygons[ii]), alpha=0.1, edgecolor='black', linewidth=3))

            # plot neighbors (useful for debugging)
            # for ii in range(self.n_agents):
            #     self.ax.text(self.x[ii,0],self.x[ii,1],str(ii))
            #     for jj in range(self.n_agents):
            #         if self.A[ii,jj]:
            #             self.ax.plot([self.x[ii,0],self.x[jj,0]],[self.x[ii,1],self.x[jj,1]],linewidth=2)
        # UPDATE VORONOI CELLS
        [p.remove() for p in reversed(self.ax.patches)]
        for ii in range(len(self.polygons)):
            self.polygon_handle[ii] = self.ax.fill(*zip(*self.polygons[ii]), alpha=0.1, edgecolor='black', linewidth=3)

        # UPDATE ROBOT AND CENTROID
        self.robot_handle.set_offsets(self.x)
        self.centroid_handle.set_offsets(self.centroids)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


############# OTHER UTILITY FUNCTIONS #################################################

def density_function_ring(ring_center, ring_radius, p):
    d_center = np.linalg.norm(p - ring_center)
    d_border = np.abs(ring_radius - d_center) ** 2
    return np.exp(-2 * d_border)


def density_function_gaussian(gaussian_mu, gaussian_sigma, p):
    # very rough implementation (to FIX)
    exponent = -((p[0] - gaussian_mu[0]) ** 2 / (2 * gaussian_sigma[0] ** 2) + (
            p[1] - gaussian_mu[1]) ** 2 / (2 * gaussian_sigma[1] ** 2))
    val = np.exp(exponent)
    return val


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def extract_extrema(polygon):
    xmin = np.min(polygon[:, 0])
    xmax = np.max(polygon[:, 0])
    ymin = np.min(polygon[:, 1])
    ymax = np.max(polygon[:, 1])

    return xmin, xmax, ymin, ymax


def get_rel_neighbors(x, all_x, neighbor_radius):
    """
    Get the relative vectors to neighbors from all_x within some radius
    TODO: test
    Parameters
    ----------
    x
    all_x
    neighbor_radius

    Returns
    -------

    """
    assert x.shape[0] == all_x.shape[1]
    rel_neighbors = all_x - x
    norms = np.linalg.norm(rel_neighbors, axis=1)
    res = rel_neighbors[norms <= neighbor_radius]
    return res
