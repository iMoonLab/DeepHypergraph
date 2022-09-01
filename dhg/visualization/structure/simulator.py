from copy import deepcopy

import numpy as np
from sklearn.metrics import euclidean_distances

from .utils import safe_div


class Simulator:

    NODE_ATTRACTION = 0
    NODE_REPULSION = 1
    EDGE_REPULSION = 2
    CENTER_GRAVITY = 3

    def __init__(self, nums, forces, centers=1, damping_factor=0.999) -> None:
        self.nums = [nums] if isinstance(nums, int) else nums

        self.node_attraction = forces.get(self.NODE_ATTRACTION, None)
        self.node_repulsion = forces.get(self.NODE_REPULSION, None)
        self.edge_repulsion = forces.get(self.EDGE_REPULSION, None)
        self.center_gravity = forces.get(self.CENTER_GRAVITY, None)

        self.n_centers = len(centers)
        self.centers = centers

        if self.node_repulsion is not None and isinstance(self.node_repulsion, float):
            self.node_repulsion = [self.node_repulsion] * self.n_centers
        if self.center_gravity is not None and isinstance(self.center_gravity, float):
            self.center_gravity = [self.center_gravity] * self.n_centers

        self.damping_factor = damping_factor

    def simulate(self, init_position, H, max_iter=400, epsilon=0.001, dt=2.0) -> None:
        """
        Simulate the force-directed layout algorithm.
        """
        position = init_position.copy()
        velocity = np.zeros_like(position)
        damping = 1.0
        for it in range(max_iter):
            position, velocity, stop = self._step(position, velocity, H, epsilon, damping, dt)
            # np.save("./tmp/position_{}.npy".format(it), position)
            # np.save("./tmp/velocity_{}.npy".format(it), velocity)
            if stop:
                break
            damping *= self.damping_factor
        return position

    def _step(self, position, velocity, H, epsilon, damping, dt):
        """
        One step of the simulation.
        """
        v2v_dist = euclidean_distances(position)
        e_center = np.matmul(H.T, position) / H.sum(axis=0).reshape(-1, 1)
        v2e_dist = euclidean_distances(position, e_center) * H
        e2e_dist = euclidean_distances(e_center)

        centers = self.centers

        force = np.zeros_like(position)
        if self.node_attraction is not None:
            f = self._node_attraction(position, e_center, v2e_dist) * self.node_attraction
            assert np.isnan(f).sum() == 0
            force += f
        if self.node_repulsion is not None:
            f = self._node_repulsion(position, v2v_dist)
            if self.n_centers == 1:
                f *= self.node_repulsion[0]
            else:
                masks = np.zeros((position.shape[0], 1))
                masks[: self.nums[0]] = self.node_repulsion[0]
                masks[self.nums[0] :] = self.node_repulsion[1]
                f *= masks
            assert np.isnan(f).sum() == 0
            force += f
        if self.edge_repulsion is not None:
            f = self._edge_repulsion(e_center, H, e2e_dist) * self.edge_repulsion
            assert np.isnan(f).sum() == 0
            force += f
        if self.center_gravity is not None:
            masks = [np.zeros((position.shape[0], 1)), np.zeros((position.shape[0], 1))]
            masks[0][: self.nums[0]] = 1
            masks[1][self.nums[0] :] = 1
            for center, gravity, mask in zip(centers, self.center_gravity, masks):
                v2c_dist = euclidean_distances(position, center.reshape(1, -1)).reshape(-1, 1)
                f = self._center_gravity(position, center, v2c_dist) * gravity * mask
                assert np.isnan(f).sum() == 0
                force += f

        force *= damping

        force = np.clip(force, -0.1, 0.1)
        position += force * dt
        velocity = force

        return position, velocity, self._stop_condition(velocity, epsilon)

    def _node_attraction(self, position, e_center, v2e_dist, x0=0.1, k=1.0):
        """
        Node attracted by edge center.
        """
        x = deepcopy(v2e_dist)
        x[v2e_dist > 0] -= x0
        f_scale = k * x  # (n, m)
        f_dir = e_center[np.newaxis, :, :] - position[:, np.newaxis, :]  # (1, m, 2) - (n, 1, 2) -> (n, m, 2)
        f_dir_len = np.linalg.norm(f_dir, axis=2)  # (n, m)
        # f_dir = f_dir / f_dir_len[:, :, np.newaxis]  # (n, m, 2)
        f_dir = safe_div(f_dir, f_dir_len[:, :, np.newaxis])  # (n, m, 2)
        f = f_scale[:, :, np.newaxis] * f_dir  # (n, m, 2)
        f = f.sum(axis=1)  # (n, 2)
        return f

    def _node_repulsion(self, position, v2v_dist, k=1.0):
        """
        Node repulsed by other nodes.
        """
        dist = v2v_dist.copy()
        r, c = np.diag_indices_from(dist)
        dist[r, c] = np.inf

        f_scale = k / (dist ** 2)  # (n, n) with diag 0
        f_dir = position[:, np.newaxis, :] - position[np.newaxis, :, :]  # (n, 1, 2) - (1, n, 2) -> (n, n, 2)
        f_dir_len = np.linalg.norm(f_dir, axis=2)  # (n, n)
        f_dir_len[r, c] = np.inf
        # f_dir = f_dir / f_dir_len[:, :, np.newaxis]  # (n, n, 2)
        f_dir = safe_div(f_dir, f_dir_len[:, :, np.newaxis])  # (n, n, 2)
        f = f_scale[:, :, np.newaxis] * f_dir  # (n, n, 2)
        f[r, c] = 0
        f = f.sum(axis=1)  # (n, 2)
        return f

    def _edge_repulsion(self, e_center, H, e2e_dist, k=1.0):
        """
        Edge repulsed by other edges.
        """
        dist = e2e_dist.copy()
        r, c = np.diag_indices_from(dist)
        dist[r, c] = np.inf

        f_scale = k / (dist ** 2)  # (m, m)
        f_dir = e_center[:, np.newaxis, :] - e_center[np.newaxis, :, :]  # (m, 1, 2) - (1, m, 2) -> (m, m, 2)
        f_dir_len = np.linalg.norm(f_dir, axis=2)  # (m, m)
        f_dir_len[r, c] = np.inf
        # f_dir = f_dir / f_dir_len[:, :, np.newaxis]  # (m, m, 2)
        f_dir = safe_div(f_dir, f_dir_len[:, :, np.newaxis])  # (m, m, 2)
        f = f_scale[:, :, np.newaxis] * f_dir  # (m, m, 2)
        f[r, c] = 0
        f = f.sum(axis=1)  # (m, 2)
        return np.matmul(H, f)

    def _center_gravity(self, position, center, v2c_dist, k=1):
        """
        Node attracted by center.
        """
        f_scale = v2c_dist  # (n, 1)
        f_dir = center[np.newaxis, np.newaxis, :] - position[:, np.newaxis, :]  # (1, 1, 2) - (n, 1, 2) -> (n, 1, 2)
        f_dir_len = np.linalg.norm(f_dir, axis=2)  # (n, 1)
        # f_dir = f_dir / f_dir_len[:, :, np.newaxis]  # (n, 1, 2)
        f_dir = safe_div(f_dir, f_dir_len[:, :, np.newaxis])  # (n, 1, 2)
        f = f_scale[:, :, np.newaxis] * f_dir  # (n, 1, 2)
        # f = jitter(f)
        f = f.sum(axis=1) * k
        return f

    def _stop_condition(self, velocity, epsilon):
        """
        Stop condition.
        """
        return np.linalg.norm(velocity) < epsilon
