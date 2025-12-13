"""
Coordinate conventions used by `bempp_audio`.

All geometry is expressed in meters in a right-handed coordinate system:

- +x: right, +y: up, +z: forward (radiation direction)
- The baffle / waveguide mouth plane is `z = 0`.
- A standalone waveguide is centered at the origin, so its mouth center is
  `(0, 0, 0)` and the throat plane is at `z = -length`.

Waveguide-on-box (unified enclosure) uses a *box frame* on the front face:

- The front face is still `z = 0`, but `x ∈ [0, box_width]` and
  `y ∈ [0, box_height]`.
- `mount_x` / `mount_y` and stored `mouth_center` are expressed in this box
  frame, with origin at the front-face lower-left corner `(0, 0, 0)`.

Polar / Plane Conventions
-------------------------
Directivity and “polar sweep” helpers use angles measured from the forward
axis (0° = on-axis). Plane selection is defined relative to the forward axis:

- `horizontal`: the sweep plane prefers the global +x direction as the lateral
  axis (i.e. it is the plane spanned by `axis` and “as close as possible to +x”).
- `vertical`: the sweep plane prefers the global +y direction as the lateral
  axis (i.e. it is the plane spanned by `axis` and “as close as possible to +y”).

For the common case where `axis == +z`, these correspond to the conventional
`xz` (horizontal) and `yz` (vertical) planes.
"""

from __future__ import annotations

from typing import Tuple

MOUTH_PLANE_Z: float = 0.0


def throat_plane_z(length: float) -> float:
    """Throat plane location for a waveguide of given length."""
    return -float(length)


def as_point3(p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3-tuple of floats."""
    return float(p[0]), float(p[1]), float(p[2])
