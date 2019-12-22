import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Body:
    """An object defining each celestial body. All Bodies exist in
    the Universe, and with each timestep, the Universe updates each
    Body's position and momentum."""

    def __init__(self, mass, initial_position, initial_velocity):
        assert isinstance(initial_position, np.ndarray)
        assert isinstance(initial_velocity, np.ndarray)

        self.mass = mass
        self.position = initial_position.astype('float64')
        self.velocity = initial_velocity.astype('float64')

    def calculate_force(self, other_bodies):
        """Calculate the gravitational force magnitude and direction
        on this body from the other bodies."""
        G = 6.67e-11 # units of m3 kg−1 s−2
        F_net = 0

        # compute force from each body
        for body2 in other_bodies:
            r12 = body2.position - self.position
            F12 = G*self.mass*body2.mass/np.linalg.norm(r12)**3 * r12
            F_net += F12

        return F_net

    def update_position(self, other_bodies, dt):
        F_net = self.calculate_force(other_bodies)
        a = F_net / self.mass # Newton's 2nd Law
        self.velocity += a * dt
        self.position += self.velocity * dt

        return self.position

class Universe:
    def __init__(self, bodies, size):
        self.bodies = bodies
        self.size = size
        self.center_of_mass = self.compute_COM(bodies)

    def compute_COM(self, bodies):
        # compute the total mass of all objects
        mass_x_pos, total_mass = np.zeros(3), 0
        for b in bodies:
            mass_x_pos += b.mass * b.position
            total_mass += b.mass

        return mass_x_pos / total_mass

    def update_timestep(self, dt):
        for i, body in enumerate(self.bodies):
            other_bodies = np.delete(bodies, i)
            body.update_position(other_bodies, dt)

        # update COM each iteration
        self.center_of_mass = self.compute_COM(self.bodies)

    def plot(self, ax):
        # get coordinates for all objects
        positions = [b.position for b in self.bodies]
        x, y, z = [p[0] for p in positions], [p[1] for p in positions], [p[2] for p in positions]

        # determine the axes limits from COM and universe size
        plt_low_lim = self.center_of_mass - self.size
        plt_high_lim = self.center_of_mass + self.size
        plt_limits = np.array([plt_low_lim, plt_high_lim]).T

        ax.clear()
        ax.scatter(x, y, z)
        ax.set(xlim=plt_limits[0], ylim=plt_limits[1], zlim=plt_limits[2])
        plt.pause(0.01)

if __name__ == "__main__":
    # define the three bodies and timestep
    body1 = Body(1e10, np.array([0, 0, 10]), np.zeros(3))
    body2 = Body(10, np.array([10, 0, 0]), np.zeros(3))
    body3 = Body(10, np.array([0, 10, 0]), np.zeros(3))

    sun = Body(1/6.67e-11, np.array([0, 0, 0]), np.array([0, -1/100, 0]))
    pebble = Body(1, np.array([100, 0, 0]), np.array([0, 1/10, 0]))

    bodies = [sun, pebble]
    timestep = 100 # seconds
    u = Universe(bodies, 5)

    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig)

    while True:
        u.update_timestep(timestep)
        u.plot(ax)
