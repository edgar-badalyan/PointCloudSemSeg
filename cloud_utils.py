import laspy
import numpy as np
import os
from skimage.io import imsave
import subprocess
import multiprocessing as mp
import json

import matplotlib as mpl
import matplotlib.cm as cm

from typing import Tuple, Callable

# Set path to data directories here
LAZ_DIR = ""
PTX_DIR = ""
IMAGE_DIR = ""

# Change to "LAZ" if using laz files
CLOUD_FORMAT = "PTX"


def ptx2laz(name: str):
    """
    Convert cloud in PTX format to LAZ format. Cloud is assumed to be in the directory PTX_DIR specified above.
    Parameters
    ----------
    name: str
    Name of the file (without extension)

    Returns
    -------

    """
    ipath = f"{PTX_DIR}/{name}.ptx"
    opath = f"{LAZ_DIR}/{name}.laz"

    # If using WSL
    command = f"wsl txt2las -i {ipath} -iptx -iparse xyzi -o {opath} -olaz"
    # Otherwise
    # command = f"txt2las -i {ipath} -iptx -iparse xyzi -o {opath} -olaz"

    print(subprocess.check_call(command.split(" ")))


def get_ptx_dims(ptx_metadata_file: str) -> dict:
    """
    Reads metadata file to get dimensions of clouds (because PTX includes the dimensions but LAZ might not).

    Parameters
    ----------
    ptx_metadata_file: str
    Path to the metadata file.

    Returns
    -------
    data: dict
    """
    with open(ptx_metadata_file) as f:
        data = json.load(f)

    return data


# Metadata file must contain shape of clouds
PTX_METADATA = get_ptx_dims("ptx_dims.json")


def ptx_data(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads data from a PTX file. Only the position of the points and the intensity is
    read: if color information is included, it is ignored.
    Parameters
    ----------
    name: str
    Name of the file (without extension). File is assumed to be in the specified directory

    Returns
    -------
    X, Y, Z, intensity: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    with open(f"{PTX_DIR}/{name}.ptx") as f:
        width = int(f.readline())
        height = int(f.readline())

        X = np.empty(shape=(height * width), dtype="float")
        Y = np.empty(shape=(height * width), dtype="float")
        Z = np.empty(shape=(height * width), dtype="float")

        intensity = np.empty(shape=(height * width), dtype="float")

        for i in range(8):
            f.readline()

        for i in range(width * height):
            line = f.readline().split(" ")

            X[i] = float(line[0])
            Y[i] = float(line[1])
            Z[i] = float(line[2])
            intensity[i] = float(line[3])

    return X, Y, Z, intensity


def laz_data(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads data from a LAZ file. Only the position of the points and the intensity is
    read: if color information is included, it is ignored.
    Parameters
    ----------
    name: str
    Name of the file (without extension). File is assumed to be in the specified directory

    Returns
    -------
    X, Y, Z, intensity: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    laz = laspy.read(f"{LAZ_DIR}/{name}.laz")

    shape = PTX_METADATA[name]

    X = np.reshape(laz.points.x, newshape=shape, order='F')
    Y = np.reshape(laz.points.y, newshape=shape, order='F')
    Z = np.reshape(laz.points.z, newshape=shape, order='F')
    intensity = np.reshape(laz.points.intensity, newshape=shape, order='F')

    return X, Y, Z, intensity


# Decide which method is used to read the cloud
if CLOUD_FORMAT == "PTX":
    READ_CLOUD = ptx_data
elif CLOUD_FORMAT == "LAZ":
    READ_CLOUD = laz_data
else:
    raise NotImplementedError


def cloud2im(name: str):
    """
    Converts a point cloud to an equirectangular image containing only the intensity of points.
    Parameters
    ----------
    name: str
    Name of the file (without extension). File is assumed to be in the specified directory
    Returns
    -------

    """
    _, _, _, intensity = READ_CLOUD(name)

    intensity = (intensity / 16).astype("uint8")

    imsave(f"{IMAGE_DIR}/{name}.png", intensity)


def cloud2im_depth(name: str):
    """
    Convert a point cloud to an equirectangular image containing the depth and intensity of points.
    Parameters
    ----------
    name: str
    Name of the file (without extension). File is assumed to be in the specified directory

    Returns
    -------

    """

    # Minimal and maximal depth allowed in the cloud
    D_MIN = 0.
    D_MAX = 100.

    X, Y, Z, intensity = READ_CLOUD(name)

    d_sq = X * X + Y * Y + Z * Z

    depth = np.sqrt(d_sq)

    # Saving the pixel coordinates of invalid depths so they can be set to a different color
    invalid_high = depth >= D_MAX
    invalid_low = depth <= D_MIN

    # Clipping the depth values between accepted bounds
    depth[depth > D_MAX] = D_MAX
    depth[depth < D_MIN] = D_MIN

    # Normalize depth between 0 and 1
    norm = mpl.colors.Normalize(vmin=D_MIN, vmax=D_MAX)

    cmap = cm.viridis
    # cmap = cm.turbo

    # Convert depth to color
    colormap = cm.ScalarMappable(norm=norm, cmap=cmap)

    colormap = colormap.to_rgba(depth, bytes=True)

    # Add intensity as alpha
    intensity = (intensity / 16).astype("uint8")

    colormap[:, :, 3] = intensity

    colormap[invalid_high] = [0, 0, 0, 0]
    colormap[invalid_low] = [0, 0, 0, 0]

    imsave(f"{IMAGE_DIR}/{name}.png", colormap, check_contrast=False)


def normalize(arr: np.ndarray, rng_min: float, rng_max: float) -> np.ndarray:
    """
    Normalize an array of numbers so that all elements lie between rng_min and rng_max. Note that
    the array will be converted to float64 data type.

    Parameters
    ----------
    arr: np.ndarray
    Array to be normalized
    rng_min: float
    Minimal value to be present in the array
    rng_max: float
    Maximal value to be present in the array

    Returns
    -------
    np.ndarray: Normalized array
    """
    arr = arr.astype("float64")
    arr -= arr.min()
    arr *= (rng_max - rng_min) / arr.max()
    arr += rng_min
    return arr


def cloud2im_pos(name: str):
    """
    Convert a point cloud to an equirectangular image containing the position and intensity of points.
    Parameters
    ----------
    name: str
    Name of the file (without extension). File is assumed to be in the specified directory

    Returns
    -------

    """
    X, Y, Z, intensity = READ_CLOUD(name)

    X = normalize(X, 0, 255).astype("uint8")
    Y = normalize(X, 0, 255).astype("uint8")
    Z = normalize(X, 0, 255).astype("uint8")

    # Add intensity as alpha
    intensity = (intensity / 16).astype("uint8")

    im = [X, Y, Z, intensity]

    im = np.transpose(im, (1, 2, 0))

    imsave(f"{IMAGE_DIR}/{name}.png", im, check_contrast=False)


def batch_conversion(cloud_dir: str, func: Callable, num_processes: int = 2, replace: bool = False):
    """
    Converts a whole directory of clouds into a given format, using multiprocessing.
    Parameters
    ----------
    cloud_dir: str
    Name of the directory where the point clouds must be read
    func: Callable
    Function or method that converts a cloud into another format
    num_processes: int
    Number of processes to be used (adapt to specs of your machine)
    replace: bool
    If True, clouds of the source dir which already exist in converted format in the destination dir will be
    converted again and thus replace the old one. If False, they will ignored.

    Returns
    -------

    """
    cloud_files = os.listdir(cloud_dir)
    cloud_files = [name.split(".")[0] for name in cloud_files]

    images = os.listdir(IMAGE_DIR)

    # Filter clouds that already exist
    if not replace:
        cloud_files = [name for name in cloud_files if f"{name}.png" not in images]

    # Change number of processes to suit your machine
    with mp.Pool(num_processes) as p:
        p.map(func, cloud_files)


if __name__ == "__main__":

    # Example usage
    batch_conversion(PTX_DIR, cloud2im_depth)
