import numpy as np
from math import sqrt, atan2, acos, pi, sin, cos
from typing import Tuple


def face2xyz(u: float, v: float, face:str) -> Tuple[float, float, float]:
    """
    Find coordinates of the point on the given face.
    Parameters
    ----------
    u: float
    Horizontal coordinate of the point (between 0 and 1)
    v: float
    Vertical coordinate of the point (between 0 and 1)
    face:
    Face name.

    Returns
    -------
    Tuple[float, float, float]: (X, Y, Z) coordinates of the point.
    """
    if face == "X+":
        return (0.5, u, v)
    elif face == "X-":
        return (-0.5, -u, v)
    elif face == "Y+":
        return (-u, 0.5, v)
    elif face == "Y-":
        return (u, -0.5, v)
    elif face == "Z+":
        return (-v, u, 0.5)
    elif face == "Z-":
        return (v, u, -0.5)


def equi2cubemap(im: np.ndarray, mask: np.ndarray) -> dict:
    """
    Projects and equirectangular image and its mask on a cubemap.
    Parameters
    ----------
    im: np.ndarray
    Image.
    mask: np.ndarray
    Mask. Must be same shape as image.

    Returns
    -------
    dict. Dictionary mapping face names to images.
    """
    H, W = im.shape[:2]

    cube = {}
    faces = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    PHI_MIN = 0.0
    PHI_MAX = (2 * H * pi) / W
    PHI_FACTOR = 1 / (PHI_MAX - PHI_MIN)
    THETA_FACTOR = 1 / (2 * pi)

    for face in faces:
        h = w = int(W / 4)
        cube_face = np.zeros(shape=(h, w), dtype=im.dtype)

        mask_face = np.zeros(shape=(h, w), dtype=mask.dtype)

        for i in range(w):
            for j in range(h):
                u, v = i / w - 0.5, 0.5 - j / h

                x, y, z = face2xyz(u, v, face)

                R = sqrt(x * x + y * y + z * z)
                theta = atan2(y, x)
                phi = acos(z / R)

                if PHI_MIN <= phi <= PHI_MAX:
                    U = theta * THETA_FACTOR
                    V = phi / pi
                    py, px = int(V * H * pi * PHI_FACTOR), int(U * W)

                    px = px if px < W else W - 1
                    py = py if py < H else H - 1

                    cube_face[j, i] = im[py, px]
                    mask_face[j, i] = mask[py, px]

        cube[face] = (cube_face, mask_face)

    return cube


def projectX(theta: float, phi: float, x_norm: float) -> Tuple[float, float, float]:
    """
    Project angles on X face.
    Parameters
    ----------
    theta
    phi
    x_norm: float
    +1.0 of X+ face, -1.0 if X- face

    Returns
    -------

    """
    x = x_norm / 2

    rho = x / (cos(theta) * sin(phi))
    y = rho * sin(theta) * sin(phi)
    z = rho * cos(phi)

    return (x, y, z)


def projectY(theta: float, phi: float, y_norm: float) -> Tuple[float, float, float]:
    """
    Project angles on Y face.
    Parameters
    ----------
    theta
    phi
    y_norm: float
    +1.0 of Y+ face, -1.0 if Y- face

    Returns
    -------

    """
    y = y_norm / 2

    rho = y / (sin(theta) * sin(phi))
    x = rho * cos(theta) * sin(phi)
    z = rho * cos(phi)

    return (x, y, z)


def projectZ(theta: float, phi: float, z_norm: float) -> Tuple[float, float, float]:
    """
    Project angles on Z face.
    Parameters
    ----------
    theta
    phi
    z_norm: float
    +1.0 of Z+ face, -1.0 if Z- face

    Returns
    -------

    """
    z = z_norm / 2

    rho = z / (cos(phi))
    x = rho * cos(theta) * sin(phi)
    y = rho * sin(theta) * sin(phi)

    return (x, y, z)


def xyz2face(x: float, y: float, z: float, face: str) -> Tuple[float, float]:
    """
    Convert X, Y, Z coordinates to image coordinates on given face.
    Parameters
    ----------
    x
    y
    z
    face

    Returns
    -------

    """
    if face == "X+":
        U = y + 0.5
        V = z + 0.5
    elif face == "Y+":
        U = 0.5 - x
        V = z + 0.5
    elif face == "X-":
        U = 0.5 - y
        V = z + 0.5
    elif face == "Y-":
        U = x + 0.5
        V = z + 0.5
    elif face == "Z+":
        U = y + 0.5
        V = 0.5 - x
    else:
        U = y + 0.5
        V = x + 0.5

    V = 1 - V

    return U, V


def cubemap2equi_point(i: int, j: int, W: int, H: int) -> Tuple[str, int, int]:
    """
    Converts the coordinates of a pixel in equirectangular image in coordinates on a cubemap face.
    Parameters
    ----------
    i: int
    Pixel horizontal coordinate.
    j: int
    Pixel vertical coordinate.
    W: int
    Image width.
    H: int
    Image height.

    Returns
    -------
    Tuple[str, int, int]: face name and pixel coordinates
    """

    PHI_MIN = 0.0
    PHI_MAX = (2 * H * pi) / W

    u = i / W
    v = j / H

    theta = 2 * pi * u
    phi = v * (PHI_MAX - PHI_MIN) + PHI_MIN

    x = cos(theta) * sin(phi)
    y = sin(theta) * sin(phi)
    z = cos(phi)

    max_dist = max(abs(x), abs(y), abs(z))
    x_norm, y_norm, z_norm = x / max_dist, y / max_dist, z / max_dist

    if x_norm == 1 or x_norm == -1:
        x, y, z = projectX(theta, phi, x_norm)

        face = "X+" if x_norm == 1 else "X-"

    elif y_norm == 1 or y_norm == -1:
        x, y, z = projectY(theta, phi, y_norm)

        face = "Y+" if y_norm == 1 else "Y-"

    else:
        x, y, z = projectZ(theta, phi, z_norm)

        face = "Z+" if z_norm == 1 else "Z-"

    U, V = xyz2face(x, y, z, face)

    py, px = int(V * W / 4), int(U * W / 4)

    px = px if px < int(W / 4) else px - 1
    py = py if py < int(W / 4) else py - 1

    return face, px, py


def cubemap2equi(cube: dict, shape: Tuple) -> np.ndarray:
    """
    Converts a cubemap to equirectangular image.
    Parameters
    ----------
    cube: dict
    Dictionary mapping face names to images
    shape: Tuple
    Image shape. First two elements must height and width, respectively

    Returns
    -------
    np.ndarray: equirectangular image.
    """
    im = np.empty(shape=shape, dtype="uint8")
    H, W = shape[:2]

    for i in range(W):
        for j in range(H):
            face, px, py = cubemap2equi_point(i, j, W, H)
            im[j, i] = cube[face][py, px]

    return im
