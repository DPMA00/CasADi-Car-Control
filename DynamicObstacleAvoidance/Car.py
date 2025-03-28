import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Car:
    def __init__(self, x0, y0, theta0, length, width, color):
        """
        Initialize the car with its initial state and dimensions.
        :param x0, y0: Initial position (m)
        :param theta0: Initial orientation (rad)
        :param length: Car length (m)
        :param width: Car width (m)
        :param color: Color for visualization
        """
        self.state = np.array([x0, y0, theta0], dtype=np.float64)  # State: [x, y, theta]
        self.length = length
        self.width = width
        self.color = color

    def dynamics(self, state, u):
        """
        Define the car's dynamics (e.g., bicycle model).
        :param state: [x, y, theta]
        :param u: [v, delta] - velocity and steering angle
        :return: State derivative [dx/dt, dy/dt, dtheta/dt]
        """
        x, y, theta = state
        v, delta = u
        dxdt = v * np.cos(theta)
        dydt = v * np.sin(theta)
        dthetadt = v * np.tan(delta) / self.length
        return np.array([dxdt, dydt, dthetadt])

    def propagate(self, u, dt):
        """
        Propagate the car's state using RK4 integration.
        :param u: Control input [v, delta]
        :param dt: Time step (s)
        """
        k1 = self.dynamics(self.state, u)
        k2 = self.dynamics(self.state + 0.5 * dt * k1, u)
        k3 = self.dynamics(self.state + 0.5 * dt * k2, u)
        k4 = self.dynamics(self.state + dt * k3, u)
        self.state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_patch(self,label):
        """
        Return a Polygon patch for visualization.
        """
        x, y, theta = self.state
        half_length = self.length / 2
        half_width = self.width / 2

        
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])

        return Polygon(corners, closed=True, color=self.color, label = label)

    def get_state(self):
        """
        Return the current state of the car.
        """
        return self.state