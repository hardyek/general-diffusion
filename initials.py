import numpy as np

def generate_u0(nx, ny, sources):

    u0 = np.zeros((nx,ny))
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    for _ in range(sources):
        x = np.random.randint(nx//4, 3*nx//4)
        y = np.random.randint(ny//4, 3*ny//4)

        amplitude = np.random.uniform(50,100)
        sigma = np.random.uniform(nx//10, nx//5)

        u0 += amplitude * np.exp(-((xx - x)**2 + (yy - y)**2)/(2 * sigma**2))

    return u0
