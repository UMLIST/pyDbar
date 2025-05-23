import numpy as np

def generate_kgrid(r: float, m: int) -> dict:
    """
    Generate square k-grid in z-space. The number of grid points
    is (2M)^2, where M = 2^m. A disk of radius r is drawn inside of
    the square [-r, r]^2. Along each axis, between the intersections
    of the disk and the axis, there are M grid points. Beyond the
    disk, there are M/2 points on each side. The step size between
    each grid point is h = (2*r)/(M-1).

    The returned object is a dictionary with keys listed below.

    Input:
    r: float - Radius of support
    m: int   - Value for M = 2^m, the number of grids generated

    Output parameters:
    grid: NDArray - [M x M] grid points
    h: float      - Step size: h = (2*r)/(m-1)
    r: float      - Radius of support
    m: int        - The power in M = 2^m grid points
    M: int        - The unextended number of grid points M = 2^m
    """
    M = 2**m # Number of (unextended) grid points
    h = (2 * r) / (M - 1) # Step size

    # We define the (extended) edge of our square as
    # s = r + (M/2)*h = r + ((M * r) / (M - 1))
    # This is explained in "D-bar Demystified"
    s = r + ((M * r) / (M - 1))

    # The extension results in 2M x 2M grids
    grid = np.zeros((2 * M, 2 * M), dtype=complex)
    x = np.arange(start=-s, stop=s + h, step=h)

    for j in range(2 * M):
        for n in range(2 * M):
            grid[j, n] = complex(x[j], x[n])

    k_grid = {
        "grid": grid,
        "h": h,
        "r": r,
        "m": m,
        "M": M
    }

    return k_grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    k = generate_kgrid(m=4, r=2)
    grid = k["grid"]
    r = k["r"]
    eps = 0.1

    print(f"Number of unextended points: {k['M']} x {k['M']}")
    print(f"Number of actual (extended) points: {grid.shape[0]} x {grid.shape[0]}")

    # Extract points in disk
    x_disk = np.real(grid[np.abs(grid) <= r + eps])
    y_disk = np.imag(grid[np.abs(grid) <= r + eps])

    # Extract points outside of disk
    x_not_disk = np.real(grid[np.abs(grid) > r + eps])
    y_not_disk = np.imag(grid[np.abs(grid) > r + eps])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x_disk, y_disk, marker=".", linestyle="", c="red")
    ax.plot(x_not_disk, y_not_disk, marker=".", linestyle="", c="blue")
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()