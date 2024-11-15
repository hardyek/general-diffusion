import numpy as np
from tqdm import tqdm

class HeatEquation:
    def __init__(self, length=50, nx=128, ny=128, alpha=127.0):
        self.alpha = alpha
        self.nx = nx
        self.ny = ny
        self.dx = length / (nx - 1)
        self.dy = length / (ny - 1)
        self.dt = min(
            self.dx**2 / (4 * self.alpha),
            self.dy**2 / (4 * self.alpha)
        )

        # Precompute Coefficients
        self.rx = self.alpha * self.dt / self.dx**2
        self.ry = self.alpha * self.dt / self.dy**2

    def anim_solve(self, u0, tmax):
        uhistory = []
        u = u0.copy()
        t = 0

        # Calculate number of iterations for tqdm
        n_iters = int(tmax / self.dt)
        step = 0
        max_history_size = n_iters // 100 + 2
        uhistory = np.zeros((max_history_size, self.nx, self.ny))
        history_index = 0

        w = np.zeros_like(u0)

        for step in tqdm(range(n_iters), desc="Solving Heat Equation"):
            np.copyto(w, u)

            # Vectorised interior points
            u[1:-1, 1:-1] = w[1:-1, 1:-1] + \
                            self.rx * (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) + \
                            self.ry * (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2])

            # Periodic boundary conditions
            u[0, :] = u[-2, :]    # Left boundary
            u[-1, :] = u[1, :]    # Right boundary
            u[:, 0] = u[:, -2]    # Bottom boundary
            u[:, -1] = u[:, 1]    # Top boundary

            # Save state periodically (e.g., every 10 steps)
            if len(uhistory) == 0 or step % 100 == 0 or t >= tmax:
                uhistory[history_index] = u.copy()
                history_index += 1

            t += self.dt

        return uhistory[:history_index]

    def solve(self, u0, tmax):
        uhistory = []
        u = u0.copy()
        t = 0

        # Calculate number of iterations for tqdm
        n_iters = int(tmax / self.dt)
        step = 0
        history_size = n_iters
        uhistory = np.zeros((history_size, self.nx, self.ny))

        w = np.zeros_like(u0)

        for step in tqdm(range(n_iters), desc="Solving Heat Equation"):
            np.copyto(w, u)

            # Vectorised interior points
            u[1:-1, 1:-1] = w[1:-1, 1:-1] + \
                            self.rx * (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) + \
                            self.ry * (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2])

            # Periodic boundary conditions
            u[0, :] = u[-2, :]    # Left boundary
            u[-1, :] = u[1, :]    # Right boundary
            u[:, 0] = u[:, -2]    # Bottom boundary
            u[:, -1] = u[:, 1]    # Top boundary

            uhistory[step] = u.copy()
            t += self.dt

        return uhistory
