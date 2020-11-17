"""Environment representing multi-agent homogenous coverage.

An environment with multiple agents, which move around in a 2-D plane. The goal is for the agents to move around and
optimally cover a region according to an importance density function.

  Typical usage example:

  env = gym.make('HomogenousCoverage-v0')
  initial_obs = env.reset()
  obs, reward, done, info = env.step(action)

"""
from enum import Enum

import gym
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
import quadpy
from scipy.spatial import Voronoi, Delaunay
from scipy.stats import multivariate_normal
from shapely.geometry import MultiPolygon, Point
import tensorflow as tf


class DensityType(Enum):
    """The representation type of the density function."""
    GAUSSIAN = 1
    RING = 2
    GMM = 3


class HomogeneousCoverageEnv(gym.Env):
    """A Gym environment for a multi-agent team solving the homogeneous sensor coverage problem.
    """
    ## Graph Properties
    # Note these are static class in order to be accessed without a particular instantiation,
    # such as when calling unpack_obs
    n_node_features = 3  # (relative distance to centroid,agent-wise reward)
    n_edge_features = 2  # relative distance to neighbor
    n_global_features = 1  # number of global graph features
    action_dim = 2  # dimension of actions per agent
    n_states = 2  # number states per agent (just position in global frame)

    # OpenAI Gym Class Metadata
    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self, n_agents=3, dt=1.0, vel_max=1.0, max_episode_length=32, sensor_radius=np.inf, robot_start_config='uniform', robot_start_gmm_dev=1.4, gmm_k=1, gmm_dev=4):
        # Environment parameters.
        self.n_agents = n_agents  # Number of agents.
        self.max_edges = n_agents ** 2  # Maximum number of posisble edges. Observation is padded to this number.
        self.vel_max = vel_max  # Max velocity.
        self.dt = dt  # Step size.
        self.max_episode_length = max_episode_length
        self.sensor_radius = sensor_radius  # Sensor disk radius. If inf, no radius limit is imposed.
        self.sensor_radii = [sensor_radius] * n_agents  # Sensor disk radius for each agent
        self.robot_start_config = robot_start_config
        self.robot_start_gmm_dev = robot_start_gmm_dev
        # Reward model.
        self._get_reward = self._get_reward_coverage

        # Observation model.
        self._get_obs = self._get_obs_v1

        # Define rectangular environment
        # TODO (Walker): Temporary hack. Replace with this self.region obtained from env_boundaries branch once merged in
        scale_factor = 10.0 / np.sqrt(2) # diameter of smallest circumscribing circle of the region
        self.bounds = scale_factor*np.array(
                                        [-1.0, 1.0, -1.0, 1.0])  # xmin, xmax, ymin, ymax
        self.region = MultiPolygon( [ ( [(self.bounds[0],self.bounds[2]),(self.bounds[0],self.bounds[3]),(self.bounds[1],self.bounds[3]),(self.bounds[1],self.bounds[2])],[] ) ] )

        self.seed()

        # Importance density function distribution.
        self.density_type = DensityType.GMM
        if self.density_type == DensityType.RING:
            self.ring_center = np.array([0, 0])
            self.ring_radius = 3.0
        elif self.density_type == DensityType.GAUSSIAN:
            self.gaussian_mu = np.array([0, 0])
            self.gaussian_sigma = [2, 2]
        elif self.density_type == DensityType.GMM:
            self.gmm_k = gmm_k
            self.gmm_dev = gmm_dev
            self.initialize_gmm_density_function()
        else:
            assert False, "Invalid density_type."

        # Area integration scheme using Gaussian quadrature.
        self.scheme = quadpy.t2.get_good_scheme(10) #
        # Rendering
        self.fig = None
        self.redraw_density_flag = True

        self.reset()
        self._set_action_space()
        self._set_observation_space()




    # Public required API of OpenAI Gym environment.

    def reset(self):
        """Reset system to a random initial state.

        Resets the system to a random initial state and recomputes the relevant parameters, such as centroid
        and proximity graph.

        Returns:
            The observation from the new initial state.
        """
        self._initialize_robots()
        # re-initialize density function
        if self.density_type == DensityType.GMM:
            self.initialize_gmm_density_function()
        self.redraw_density_flag = True

        self._update_cached_observables()
        self.ep_length = 0
        dict_obs = self._get_obs()
        flat_obs = pack_graphs(dict_obs, self.n_edge_features, max_edges=self.max_edges)
        return flat_obs

    def step(self, unshaped_action):
        """Apply the unshaped_action to the system for 1 timestep.

        Args:
            unshaped_action: (n_agents*actions_dim,) array of actions for each agent.
                Needs to be unraveled to match the shape (n_agents, actions_dim)
        Returns:
            A tuple of (obs, reward, done, info).
            obs (array) is the observation of the current state.
            reward (float) is the reward given the last action and state.
            done (bool) is whether the task is done.
            info (dict) is a dictionary of various information to convey back to the simulation.
        """

        assert not np.isnan(unshaped_action).any(), "Actions must not be NaN."

        # Update task dynamic state (eg. agent locations).
        action = self._unpack_actions(unshaped_action)
        self._update_state(action)
        # Update cached observable quantities (eg. coverage metrics).
        self._update_cached_observables()

        # Generate observations in observation space.
        dict_obs = self._get_obs()
        flat_obs = pack_graphs(dict_obs, self.n_edge_features, max_edges=self.max_edges)

        reward = self._get_reward()

        # Check termination.
        # Terminate when the maximum episode length is reached.
        self.ep_length += 1
        done = self.ep_length >= self.max_episode_length

        # Save debugging info.
        info = {}

        # Postfix assertions.
        assert not np.isnan(flat_obs).any(), "Observations must not be NaN."
        assert not np.isnan(reward), "Reward must not be NaN."

        return flat_obs, reward, done, info

    def render(self, mode='human'):
        """Render a plot of the agents along with their Voronoi cells and centroids.

        Args:
            kwargs: Not used.

        Returns:
            None.
        """

        if mode == 'human':
            plt.ion()

        draw_sensor_disk = lambda j: self.ax.add_patch(
            plt.Circle(self.x[j, :], radius=self.sensor_radii[j], fill=True, facecolor='#A7D0B0', edgecolor='#4F61A1',
                       linewidth=1, alpha=0.2))
        cs = plt.cm.magma(np.linspace(0, 1, self.n_agents))
        draw_polygon = lambda j: self.ax.fill(*zip(*self.cell_polygons[j]), alpha=0.1, fill=False, color=cs[j], edgecolor='black',
                                               linewidth=3)
        # Content data limits in data units.
        data_xlim = [self.bounds[0] - 0.5, self.bounds[1] + 0.5]
        data_ylim = [self.bounds[2] - 0.5, self.bounds[3] + 0.5]
        data_aspect_ratio = (data_xlim[1] - data_xlim[0]) / (data_ylim[1] - data_ylim[0])


        if self.fig == None:

            # Aesthetic parameters.
            # Figure aspect ratio.
            fig_aspect_ratio = 16.0/9.0 # Aspect ratio of video.
            fig_pixel_height = 1080     # Height of video in pixels.
            dpi = 300                   # Pixels per inch (affects fonts and apparent size of inch-scale objects).

            # Set the figure to obtain aspect ratio and pixel size.
            fig_w = fig_pixel_height / dpi * fig_aspect_ratio # inches
            fig_h = fig_pixel_height / dpi # inches
            self.fig, self.ax = plt.subplots(1, 1,
                figsize=(fig_w, fig_h),
                constrained_layout=True,
                dpi=dpi)

            # Set axes limits which display the workspace nicely.
            self.ax.set_xlim(data_xlim[0], data_xlim[1])
            self.ax.set_ylim(data_ylim[0], data_ylim[1])

            # Setting axis equal should be redundant given figure size and limits,
            # but gives a somewhat better interactive resizing behavior.
            self.ax.set_aspect('equal')

            # Draw robots
            self.robot_handle = self.ax.scatter(self.x[:, 0], self.x[:, 1], 20, 'black')
            self.centroid_handle = self.ax.scatter(self.density_centroid[:, 0], self.density_centroid[:, 1], 20, marker='s',
                                                   color='green')
            self.robot_centroid_edge_artists = [self.ax.plot([x1, x2], [y1, y2], linewidth=0.2, color='gray', zorder=20)[0] for (x1, x2, y1, y2) in zip(self.x[:, 0], self.density_centroid[:, 0], self.x[:, 1], self.density_centroid[:,1])]

            # Draw voronoi cell interior
            self.polygon_handle = []
            for ii in range(len(self.cell_polygons)):
                self.polygon_handle.append(draw_polygon)

            # plot neighbors (useful for debugging)
            # for ii in range(self.n_agents):
            #     self.ax.text(self.x[ii,0],self.x[ii,1],str(ii))
            #     for jj in range(self.n_agents):
            #         if self.proximity_graph[ii,jj]:
            #             self.ax.plot([self.x[ii,0],self.x[jj,0]],[self.x[ii,1],self.x[jj,1]],linewidth=2)

            # Draw step number.
            self.step_number_artist = Annotation(text='step 0', xy=(10,10),
                xycoords='axes points')
            self.ax.add_artist(self.step_number_artist)

        if self.redraw_density_flag:
            # clear old density func if it exists
            if hasattr(self, 'density_artist'):
                for c in self.density_artist.collections:
                    c.remove()
            # Draw the density function
            ngrid = 100
            xi = np.linspace(data_xlim[0], data_xlim[1], ngrid)
            yi = np.linspace(data_ylim[0], data_ylim[1], ngrid)
            xi, yi = np.meshgrid(xi, yi)
            zi = [self.get_density_value([x, y]) for x, y in zip(xi, yi)]
            self.density_artist = plt.contourf(xi, yi, zi, levels=30, antialiased=True, cmap=plt.cm.Purples, alpha=0.4)
            self.redraw_density_flag = False

        # UPDATE VORONOI CELLS
        [p.remove() for p in reversed(self.ax.patches)]
        for ii in range(len(self.cell_polygons)):
            self.polygon_handle[ii] = draw_polygon(ii)
        for ii in range(self.n_agents):
            if self.sensor_radii[ii] != np.inf:
                draw_sensor_disk(ii)
            # TODO(Rebecca): write out the coverage for the robot
            # self.ax.annotate(str())
        # UPDATE ROBOT AND CENTROID
        self.robot_handle.set_offsets(self.x)
        self.centroid_handle.set_offsets(self.density_centroid)

        for (artist, robot, centroid) in zip(self.robot_centroid_edge_artists, self.x, self.density_centroid):
            artist.set_xdata((robot[0], centroid[0]))
            artist.set_ydata((robot[1], centroid[1]))

        # Update step number.
        self.step_number_artist.set_text(f'step {self.ep_length}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if mode == 'rgb_array':
            s, (width, height) = self.fig.canvas.print_to_buffer()
            rgb = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            return rgb


    def close(self):
        pass




    # Observation model alternatives.

    def _get_obs_v1(self):
        """
        An observation model incorporating:
            Node data:
                target density centroid of local cell
                target density integral of local cell
            Edge data:
                displacement vector to cell neighbors

        Returns:
            A dictionary of the observation. It contains nodes, edges, and
            globals. See the wiki for a more complete definition.
        """
        node_obs = self._node_obs_density()
        edge_obs = self._edge_obs_cell_neighbor_displacement()
        global_obs = np.zeros((1, self.n_global_features))
        dict_obs = {
            "nodes": node_obs,
            "edges": edge_obs,
            "globals": global_obs,
        }
        return dict_obs

    def _node_obs_centroid(self):
        """ A node observation model incorporating the displacement vector to
        the target density function weighted cell centroid. """
        feature_scale = 1.0
        rel_centroids = self.density_centroid - self.x
        node_obs = feature_scale * rel_centroids
        return node_obs

    def _node_obs_density(self):
        """ A node observation model incorporating the displacement vector to the
        target density function weighted cell centroid and the zeroth moment of
        the target density function in the cell. """
        feature_scale = 1.0
        rel_centroids = self.density_centroid - self.x
        node_obs =  feature_scale * np.concatenate([
                                            rel_centroids,
                                            np.reshape(self.density_zeroth_moment, (-1,1))],
                                            axis=1)

        return node_obs

    def _node_obs_coverage(self):
        """ A node observation model incorporating the displacement vector to the
        target density function weighted cell centroid, the zeroth moment of the
        target density function in the cell, and the integral of the point-wise
        product of the target density function and coverage cost function in the
        local cell.
        """
        raise NotImplementedError

    def _edge_obs_cell_neighbor_displacement(self):
        """ An edge observation model incorporating the displacement vector to cell neighbors.

        Returns:
            list of dicts, one for each category of edges:
            edges, edge data for each robot-robot edge
            senders, index of sender robot
            receivers, index of receiver robot
        """
        adjacency = self.compute_cell_adjacency_graph()
        (senders, receivers) = np.nonzero(adjacency)
        edge_data = self.x[senders, :] - self.x[receivers, :]

        edges_dict = []
        for (s, r, d) in zip(senders, receivers, edge_data):
            edges_dict.append({'sender': s, 'receiver': r, 'data': d})

        return edges_dict




    # Reward function alternatives.

    def _get_reward_coverage_rate(self):
        """ Reward based on the rate of change of the team coverage cost. """
        reward_scale = 1.0
        if hasattr(self, 'prev_team_coverage_cost'):
            reward = reward_scale * (self.prev_team_coverage_cost - self.team_coverage_cost) / self.dt
        else:
            reward = 0
        self.prev_team_coverage_cost = self.team_coverage_cost
        return reward

    def _get_reward_coverage(self):
        """ Reward based on the team coverage cost. """
        reward_scale = 0.2
        reward = -reward_scale * self.team_coverage_cost
        return reward

    def _get_reward_centroid_distance(self):
        """ Reward based on sum of distances to centroids of the containing cells. """
        reward = np.sum(np.linalg.norm(self.x - self.density_centroid, axis=1))
        return




    # State initialization and update.

    def _initialize_robots(self):
        """
        Initializes the robots (self.x, a n_agentsx2 numpy array) based on the robot_start_config parameter. Supported
        options are:
            'uniform': uniformly distributes the robots in the boundary polygon
            'line': uniformly distributes the robots along a line across the boundary region. The line has random slope
                and y-intercept. The robots are perturbed slightly from the line to avoid collinear errors
            'gmm': robots are placed based on a gaussian mixture model, with randomly selected parameters (number of means,
                location of means, covariances)
        """
        #dummy code for compatibility with master. TODO delete when multipolygon boundary representation is merged into master
        self.region = MultiPolygon( [ ( [(self.bounds[0],self.bounds[2]),(self.bounds[0],self.bounds[3]),(self.bounds[1],self.bounds[3]),(self.bounds[1],self.bounds[2])],[] ) ] )

        if self.robot_start_config == 'uniform': #default
            self.x = _sample_polygon_uniform(self.region, self.n_agents)

        if self.robot_start_config == 'line':
            meanx = np.random.uniform(self.region.bounds[0]*0.5, self.region.bounds[2]*0.5)
            meany = np.random.uniform(self.region.bounds[1]*0.5, self.region.bounds[3]*0.5)

            unit_length = np.sqrt(self.region.area)/2 #characteristic of radius

            #shift points from horizontal line to line with slope
            slope = np.random.uniform(-5,5)

            points = []
            while len(points) < self.n_agents:
                px = np.random.uniform(meanx-3*unit_length, meanx+3*unit_length)
                py = np.random.normal(0,0.05)
                p = np.array([px, py]) + np.array([meany, slope*(px - meanx)]) # adjust point according to line slope
                if self.region.contains(Point(p[0], p[1])): # if selected robot position is in bounds
                    points.append(p)
            self.x = np.array(points)

        if self.robot_start_config == 'gmm':
            k = np.random.randint(1,np.floor(self.n_agents/3)+1)
            #generate list of k gaussians
            gaussians = []
            gaussian_means = _sample_polygon_uniform(self.region, k)
            for i in range(k):
                mean = gaussian_means[i]
                gmm_dev_rand = np.random.normal(self.gmm_dev ** 2, self.gmm_dev * 0.1)
                cov = np.eye(2)*gmm_dev_rand**2
                gaussians.append( multivariate_normal(np.array(mean), cov) ) # multivariate normal dist object

            # sample gaussian from list, sample agent position from gaussian
            points = []
            while len(points) < self.n_agents:
                dist_ind = np.random.randint(0, k)
                dist = gaussians[dist_ind]
                p = dist.rvs()
                if self.region.contains(Point(p[0], p[1])): # if selected robot position is in bounds
                    points.append(p)

            self.x = np.array(points)
        return

    def _update_state(self, action):
        """Apply the action to the dynamics of the system and update the state.

        It is a simple integration of the action treated as the velocity of the agent.
        It also updates the state of the proximity graph and the centroids.

        Args:
            action: An array vx, vy of the velocity to be applied in the x and y directions.

        Returns:
            None
        """
        unbounded_x = self.x + self.dt * action

        # Force agents to remain in bounds. Presently Voronoi assumptions can't
        # yet handle agents on the boundary or collinear agents. The stupid hack
        # fix here is to constrain agents to eps inside of bounds, where the
        # value eps is different for each agent.
        lower = np.reshape(self.bounds[[0,2]], (1,2))
        upper = np.reshape(self.bounds[[1,3]], (1,2))
        eps = 0.1 + 0.1 * np.random.random_sample((self.n_agents, 2))
        self.x = np.clip(unbounded_x, lower+eps, upper-eps)




    # Coverage and Voronoi cell computation helpers.

    def _update_cached_observables(self):
        # Calculate coverage metrics.
        self.cell_polygons, self.cell_triangulations = self.compute_voronoi_cells()
        (self.density_zeroth_moment, self.density_first_moment, self.density_centroid) = self.compute_density_metrics()
        agent_coverage_cost = self.compute_coverage_metrics()
        self.team_coverage_cost = np.sum(agent_coverage_cost)


    def compute_voronoi_cells(self):
        """Compute a bounded version of the Voronoi tesselation of the robots.

        Computes a set of finite Voronoi cells by creating 'dummy' robots mirrored on the environment boundary.
        Returns a list of numpy arrays signifying the vertices of n_agents Voronoi polygons

        Returns:
            A list of vertices consisting of n_agents polygons
        """

        # Generate Voronoi cell polygons.
        # Create reflections of robot positions along all the environment boundaries.
        #(TODO: Sid) Make this work for general polygons
        points_center = np.copy(self.x)
        points_left = np.copy(points_center)
        points_left[:,0] = self.bounds[0] - (points_left[:, 0] - self.bounds[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = self.bounds[1] + (self.bounds[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = self.bounds[2] - (points_down[:, 1] - self.bounds[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = self.bounds[3] + (self.bounds[3] - points_up[:, 1])
        points = np.concatenate((points_center,points_left,points_right,points_down,points_up),axis=0)

        vor_  = Voronoi(points)                                                             # larger Voronoi tesselation
        filtered_region_indices = vor_.point_region[:vor_.npoints//5].tolist()              # we only need the first fifth of regions
        filtered_vertex_indices = [vor_.regions[ind] for ind in filtered_region_indices]    # get vertices of those regions
        cell_polygons = [vor_.vertices[index_set] for index_set in filtered_vertex_indices]

        # Generate triangulations of cells for use in integration.
        cell_triangulations = []
        for p in cell_polygons:
            hull = Delaunay(p)
            cell_triangulations.append(p[hull.simplices.T])

        return cell_polygons, cell_triangulations

    def compute_density_metrics(self):
        """Compute metrics of the target density function local to each agent.

        Computes the zeroth moment, first moment, and centroid of the density
        function within each agent's cell.

        Returns:
            density_zeroth_moment, (n,)
            density_first_moment, (n,2)
            density_centroid, (n,2)
        """

        density_zeroth_moment = np.zeros(self.n_agents)
        density_first_moment  = np.zeros((self.n_agents, 2))
        density_centroid      = np.zeros((self.n_agents, 2))
        for ii in range(self.n_agents):
            density_first_moment[ii,:] = np.sum(
                self.scheme.integrate(lambda p: p*self.get_density_value_of_agent_i(p, ii), self.cell_triangulations[ii]), axis=1)
            if np.linalg.norm(density_first_moment[ii,:]) < 1.0e-15:
                density_first_moment[ii,:] = self.x[ii,:] # first moment is zero--set it as robot location

            density_zeroth_moment[ii] = np.sum(
                self.scheme.integrate(lambda p: self.get_density_value_of_agent_i(p, ii), self.cell_triangulations[ii]))

        denominator = np.reshape(np.where(density_zeroth_moment > 1.0e-15, density_zeroth_moment, 1), (-1,1))
        density_centroid = density_first_moment / denominator
        return (density_zeroth_moment, density_first_moment, density_centroid)

    def compute_coverage_metrics(self):
        """Compute metrics of the achieved coverage.

        Computes the integral of the pointwise product of the target density
        function and the agent coverage quality function within each agent's
        cell.
        """
        agent_coverage_cost = np.zeros(self.n_agents)
        for ii in range(self.n_agents):
            agent_coverage_cost[ii] = np.sum(self.scheme.integrate(lambda p: self.coverage_quality(p,ii), self.cell_triangulations[ii]))
        return agent_coverage_cost

    def coverage_quality(self,p,ii):
        """Calculates the quality of coverage/surveillance provided by a robot.
        This is a function of the euclidean distance between the robot's
        location (x) and the points of interest (p), scaled by an
        importance function.
        Args:
            p: (state_dim,num_triangles_being_integrated,num_integration_points)
            x: (state_dim)
        Returns:
            (num_triangles_being_integrated,num_integration_points) floats representing
                    the coverage quality at evaluation points.
        """
        x_minus_q = (self.x[ii,:] - p.T).T
        return np.sqrt(np.sum(x_minus_q**2.0,axis=0))*self.get_density_value_of_agent_i(p, ii)

    def get_density_value_of_agent_i(self, p, i):
        """Gets the density value at point p given agent of index i If p is outside of the sensing radius of agent i,
        0 is returned.

        Args:
            p: An array corresponding to the x,y position to get the density value of.
               The shape is (state_dim,num_triangles_being_integrated,num_integration_points)
            i: An integer corresponding to the index of the agent

        Returns:
            A float as the density value at point p.
        """
        unbounded_densities = self.get_density_value(p)
        agent_p = self.x[i]  # agent i's position
        agent_sensor_radius = self.sensor_radii[i]  # agent i's sensor radius
        # We assume p[:, j, k] is a single point, so find the radius based off of that
        distances = np.linalg.norm(p.T - agent_p, axis=2).T  # shape = (num_triangles_being_integrated,num_integration_points)
        are_within_radius = distances <= agent_sensor_radius
        # Multiply densities outside of boundary with 0.
        densities = are_within_radius.astype(float) * unbounded_densities
        return densities

    def get_density_value(self, p):
        """Gets the density value at point p.

        Args:
            p: An array corresponding to the x,y position to get the density value of

        Returns:
            A float as the density value at point p.

        Raises:
            Exception if self.DensityType is not known.
        """
        if self.density_type == DensityType.GAUSSIAN:
            return density_function_gaussian(self.gaussian_mu, self.gaussian_sigma, p)
        elif self.density_type == DensityType.RING:
            return density_function_ring(self.ring_center, self.ring_radius, p)
        elif self.density_type == DensityType.GMM:
            return density_function_gmm(self.gmm_means, self.gmm_covariances, p)
        else:
            raise Exception('Unknown DensityType %s' % str(self.density_type))

    def initialize_gmm_density_function(self):
        """Initializes parameters self.gmm_means, self.gmm_covariances for use in calculating gmm density function
        """
        # dummy code for compatibility with master. TODO delete when multipolygon boundary representation is merged into master
        self.region = MultiPolygon([([(self.bounds[0], self.bounds[2]), (self.bounds[0], self.bounds[3]),
                                      (self.bounds[1], self.bounds[3]), (self.bounds[1], self.bounds[2])], [])])

        self.gmm_means = _sample_polygon_uniform(self.region, self.gmm_k) #kx2 matrix of gmm means chosen from within bounds
        self.gmm_covariances = []
        for i in range(self.gmm_k):
            # select covariance for the ith gaussian, choosing from a normal distribution centered at the input
            # std dev squared, with a std dev of 0.1* the input std dev. i.e. input 1.4 --> normal distribution centered
            # at 2 with std dev 0.14
            cov_constant = np.random.normal(self.gmm_dev**2, self.gmm_dev*0.1)
            cov = [cov_constant, cov_constant]
            self.gmm_covariances.append(cov)  #list of variances of two gaussians


    # Edge relationship helpers.

    def have_common_vertex(self, ii, jj):
        """Checks if the common vertices between the Voronoi cells of agent ii and jj lie within the domain.

        Args:
            ii is the index of the first agent.
            jj is the index of the second agent.

        Returns:
            A boolean
        """

        common_vertices = np.array(
            [x for x in set(tuple(x) for x in self.cell_polygons[ii]) & set(tuple(x) for x in self.cell_polygons[jj])])
        has_common_vertex_in_domain = False
        for jj in range(common_vertices.shape[0]):
            if self.bounds[0] <= common_vertices[jj, 0] <= self.bounds[1] and self.bounds[2] <= common_vertices[
                jj, 1] <= self.bounds[3]:
                has_common_vertex_in_domain = True
                break
        return has_common_vertex_in_domain

    def compute_cell_adjacency_graph(self):
        """Compute the proximity graph based on the Delaunay triangulation.

        Agents i and j are neighbors if they share a Voronoi edge. This information is then stored in a proximity
        graph, or adjacency matrix.

        Returns:
            The proximity graph as an unweighted adjacency matrix.
        """
        # compute the DeLaunay triangulation
        triangulation = Delaunay(self.x)
        # get the neighbors
        helper = triangulation.vertex_neighbor_vertices
        index_pointers = helper[0]
        indices = helper[1]
        # construct the adjacency matrix
        proximity_graph = np.zeros((self.n_agents, self.n_agents))
        for ii in range(self.n_agents):
            proximity_graph[ii, ii] = 1  # adding a self loop (do we need this?)
            neigh_list = indices[index_pointers[ii]:index_pointers[ii + 1]]
            for jj in neigh_list:
                if self.have_common_vertex(ii, jj):
                    proximity_graph[ii, jj] = 1

        return proximity_graph




    # Misc helper functions.

    def _set_action_space(self):
        """ Set the fixed action space based on the number of agents. """
        self.action_space = gym.spaces.Box(
            shape=(self.n_agents * self.action_dim,),
            low=-self.vel_max,
            high=self.vel_max,
            dtype=np.float32)

    def _set_observation_space(self):
        """ Set the fixed observation space based on the observation function. """
        dummy_dict_obs = self._get_obs()
        dummy_flat_obs = pack_graphs(dummy_dict_obs, self.n_edge_features, max_edges=self.max_edges)
        inf = np.float32(np.Inf)
        self.observation_space = gym.spaces.Box(shape=(dummy_flat_obs.size,), low=-inf, high=inf, dtype=np.float32)


    def _unpack_actions(self, flat_action):
        """ Unpack flat action representation and clip to domain.

        Inputs
            flat_action, (2*n,)
        Returns
            action, (n,2)
        """
        action = np.reshape(flat_action, (self.n_agents, self.action_dim))
        action_norm = np.linalg.norm(action, axis=1)
        scale_mask = action_norm > self.vel_max

        normed_vals = self.vel_max*np.divide( np.multiply(action.T, scale_mask), action_norm).T
        normed_vals[np.isnan(normed_vals)] = 0.0
        action = np.multiply(action.T,(action_norm<=self.vel_max)).T + normed_vals
        return action

    def expert_action(self):
        """ Return the Lloyd's algorithm expert velocities for each agent.

        These are scaled displacement vectors from each agent to its Voronoi
        cell centroid. The centroids must have been updated prior to calling.
        This function is used for training dataset generation.

        Returns:
            vel, (n,2) velocity vector of n agent
        """
        rel_centroids = self.density_centroid - self.x
        # Lloyd's algorithm: Drive to the centroid
        vel = 4.0 * rel_centroids
        return vel

    @staticmethod
    def unpack_obs(obs, ob_space):
        """Unpacks the observation from a flat array into all of the objects needed to construct the observation graph.

        Unpacks the observation, masking off invalid edges. Invalid edges are where the sender value is equal to -1.
        If multiple observations are passed in, the observations are concatenated together, appropriately updating the
        indexing of the senders and receivers.

        Derived from https://github.com/katetolstaya/gym-flock/blob/experimental/gym_flock/envs/spatial/coverage.py

        Args:
            obs: An array of observations, all of which are flat arrays.
            ob_space: The observation_space (gym.spaces.Box) of the environment.

        Returns:
            A tuple of batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs.
            batch_size(int): the size of the batch to be run through a model.
            n_node(Tensor): Tensor of the number of nodes in each batch. Shape=(batch_size,).
            nodes(Tensor): Tensor of the nodes in the observation. Shape=(batch_size * n_agents, n_node_features)
            n_edge(Tensor): Tensor of the number of edges in each batch. Shape=(batch_size,)
            edges(Tensor): Tensor of the edges in the observation. Shape=(batch_size*number of edges, n_edge_features)
            senders(Tensor): Tensor of the senders in the observation. Shape=(batch_size*number of edges,)
            receivers(Tensor): Tensor of the senders in the observation. Shape=(batch_size*number of edges,)
            globs(Tensor): Tensor of the global values of the observation. Shape=(batch_size,n_global_features)
        """

        assert tf is not None, "Function unpack_obs() is not available if Tensorflow is not imported."

        nodes_w = HomogeneousCoverageEnv.n_node_features
        edges_w = HomogeneousCoverageEnv.n_edge_features
        globals_w = HomogeneousCoverageEnv.n_global_features
        # assume max edges is n**2
        discriminant = nodes_w ** 2 - 4 * (2 + edges_w) * (globals_w - ob_space.shape[0])
        assert discriminant >= 0.0, "Error in calculation of number of nodes."
        nodes_n = int((-nodes_w + discriminant ** 0.5) / (2 * (2 + edges_w)))
        edges_n = nodes_n ** 2
        nodes_shape = (nodes_n, nodes_w)
        edge_data_shape = (edges_n, edges_w)
        senders_receivers_shape = (edges_n, 1)
        globals_shape = (1, globals_w)

        shapes = (globals_shape, nodes_shape, senders_receivers_shape, senders_receivers_shape, edge_data_shape)
        sizes = [np.prod(s) for s in shapes]
        assert sum(sizes) == obs.shape[1], "Observation shape is wrong, expected %d and got %s" % (
            sum(sizes), str(obs.shape))
        tensors = tf.split(obs, sizes, axis=1)
        tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
        (globs, nodes, senders, receivers, edges) = tensors
        batch_size = tf.shape(nodes)[0]
        nodes = tf.reshape(nodes, (-1, nodes_w))
        n_node = tf.fill((batch_size,), nodes_n)  # assume n nodes is fixed

        # compute edge mask and number of edges per graph
        mask = tf.reshape(tf.not_equal(senders, -1), (batch_size, -1))  # padded edges have sender = -1
        n_edge = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
        mask = tf.reshape(mask, (-1,))

        # generate edge index offset to offset edges
        # the index offset should be equal to the number of nodes, so it references the second, third, fourth graph etc.
        index_offset = tf.cumsum(n_node, exclusive=True)  # exclusive makes it such that we start with 0, not n_edge[0]
        index_offset = tf.reshape(tf.cast(index_offset, senders.dtype),
                                  (-1, 1, 1))  # reshape to be sender/receiver shape
        senders = tf.add(senders, index_offset)
        receivers = tf.add(receivers, index_offset)

        # flatten edge data
        edges = tf.reshape(edges, (-1, edges_w))
        senders = tf.reshape(senders, (-1,))  # remove extra dimension since feature width is 1
        receivers = tf.reshape(receivers, (-1,))

        # mask edges
        edges = tf.boolean_mask(edges, mask, axis=0)
        senders = tf.boolean_mask(senders, mask)
        receivers = tf.boolean_mask(receivers, mask)

        globs = tf.reshape(globs, (batch_size, globals_w))

        # cast all indices to int
        n_node = tf.cast(n_node, tf.int32)
        n_edge = tf.cast(n_edge, tf.int32)
        senders = tf.cast(senders, tf.int32)
        receivers = tf.cast(receivers, tf.int32)

        return batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs,


############# OTHER UTILITY FUNCTIONS #################################################

def density_function_ring(ring_center, ring_radius, p):
    """Calculates the density at point p for a density function that is a ring.

    Args:
        ring_center: An array of the x,y coordinates of the center of the ring.
        ring_radius: A float of the radius of the ring.
        p: The point to evaluate the density function at.

    Returns:
        The value of the density function at point p.
    """
    d_center = np.linalg.norm(p - ring_center)
    d_border = np.abs(ring_radius - d_center) ** 2
    return np.exp(-2 * d_border)


def density_function_gaussian(gaussian_mu, gaussian_sigma, p):
    """Calculates the density at point p for a density function that is a gaussian.

    Args:
        gaussian_mu: An array of the x,y coordinates of the mean of the gaussian
        gaussian_sigma: An array of the x,y coordinates of the variances of the gaussian
        p: The point to evaluate the density function at.

    Returns:
        The value of the density function at point p.
    """
    # very rough implementation (to FIX)
    exponent = -((p[0] - gaussian_mu[0]) ** 2 / (gaussian_sigma[0] ** 2) + (
            p[1] - gaussian_mu[1]) ** 2 / (gaussian_sigma[1] ** 2))
    val = 1.0/(2 * np.pi * gaussian_sigma[0] * gaussian_sigma[1]) * np.exp(exponent)
    return val

def density_function_gmm(gmm_means, gmm_covs, p):
    """Calculates the density at point p for a density function that is a gaussian.

    Args:
        gmm_means: A list of numpy arrays of the x,y coordinates of the means of the k gaussians that make up the gmm
        gmm_covs: A list of variance pairs [var_x, var_y] of the x,y coordinates of the k gaussians that make up the gmm
        p: The point to evaluate the density function at.

    Returns:
        The value of the density function at point p.
    """
    # very rough implementation (to FIX)
    k = len(gmm_covs)
    val = 0.0
    for i in range(k):
        val = val + (1.0/k)*density_function_gaussian(gmm_means[i], np.sqrt(gmm_covs[i]), p)

    return val


def _get_rel_neighbors(x, all_x, neighbor_radius):
    """
    Get the relative vectors to neighbors from all_x within some radius
    TODO: test
    Parameters
    ----------
    x
    all_x:
    neighbor_radius

    Returns
    -------

    NB: Function not used for anything.
    """
    assert x.shape[0] == all_x.shape[1]
    rel_neighbors = all_x - x
    norms = np.linalg.norm(rel_neighbors, axis=1)
    res = rel_neighbors[norms <= neighbor_radius]
    return res


def pack_graphs(obs, n_edge_features, max_edges=None):
    """Packs a dictionary observation into a flat array, padding with the appropriate number of edges.

    If the number of edges in the observation are less than max_edges, then edges are added on to make the number of
    edges max_edges. These edges are invalid, and signify that by having a sender of -1. The dicitonary is then
    flattened.

    Args:
        obs: The observation as a dictionary. The dictionary should have keys "nodes", "edges", and "globals".
            obs['edges'] should be a list of dictionaries whose keys are "sender", "receiver", and "data.
        n_edge_features (int): the number of features in an edge.
        max_edges (optional), the max number of edges the observation should be padded to. If none are specified,
            no edges are added as padding.
    Returns:
        obs, flat array of observations
    Raises:
        AssertionError if the number of edges in obs['edges'] exceeds max_edges.

    """
    nodes = obs['nodes']
    edges_dict = obs['edges']
    globs = obs['globals']
    # pad edges to max_edges with pad_edge, which has sender and receiver of -1.
    if max_edges is not None:
        n_pad = max_edges - len(edges_dict)
        assert n_pad >= 0, "Number of edges exceeds max_edges!"
        pad_edge = {'data': np.zeros(n_edge_features), 'sender': -1, 'receiver': -1}
        pad_edges = [pad_edge] * n_pad
        edges_dict = edges_dict + pad_edges

    senders = np.array([e['sender'] for e in edges_dict])
    receivers = np.array([e['receiver'] for e in edges_dict])
    data = np.array([e['data'] for e in edges_dict])

    # TODO: fix the order of this somewhere so that we never pack and unpack out of order
    sources = ([globs] + [nodes] + [senders] + [receivers] + [data])
    obs = np.concatenate([d.flatten() for d in sources])
    return obs

def _sample_polygon_uniform(poly, n):
    """
    Samples n points from a shapely Polygon or MultiPolygon object

    Args:
        poly: shapely Polygon or MultiPolygon object
        n: number of points to sample

    Returns:
        an nx2 numpy array of points within the polygon
    """
    points = []
    while len(points) < n:
        px = np.random.uniform(poly.bounds[0],poly.bounds[2]) #xmin,xmax
        py = np.random.uniform(poly.bounds[1],poly.bounds[3]) #ymin,ymax
        p = np.array([px, py])
        if poly.contains(Point(p[0], p[1])):  # if selected robot position is in bounds
            points.append(p)

    return np.array(points)
