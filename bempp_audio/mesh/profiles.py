"""Reusable waveguide profile functions (pure geometry)."""

from __future__ import annotations

from typing import Optional
import warnings
import numpy as np


def conical_profile(x: np.ndarray, throat_r: float, mouth_r: float, length: float) -> np.ndarray:
    """Linear conical expansion: r(x) = r_throat + (r_mouth - r_throat) * x/L."""
    return throat_r + (mouth_r - throat_r) * (x / length)


def exponential_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    flare_constant: Optional[float] = None,
) -> np.ndarray:
    """Exponential expansion: r(x) = r_throat * exp(m*x); m auto-fits mouth if None."""
    if flare_constant is None:
        if mouth_r <= throat_r:
            raise ValueError("mouth_r must be > throat_r for exponential profile")
        flare_constant = np.log(mouth_r / throat_r) / length
    return throat_r * np.exp(flare_constant * x)


def sqrt_ease_profile(x: np.ndarray, throat_r: float, mouth_r: float, length: float) -> np.ndarray:
    """Legacy "tractrix-like" easing curve (not a true tractrix horn).

    This is a simple ease-out mapping:
        r(x) = r0 + (r1-r0) * sqrt(2t - t^2), with t=x/L

    Notes
    -----
    Historically this project called this a "tractrix" profile. A true tractrix horn
    contour is defined by a tractrix curve with a different geometric meaning.
    Prefer `tractrix_horn_profile` when you want an actual tractrix-based contour.
    """
    t = np.asarray(x, dtype=float) / float(length)
    return throat_r + (mouth_r - throat_r) * np.sqrt(np.clip(2 * t - t**2, 0.0, 1.0))


def tractrix_horn_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    *,
    n_table: int = 4096,
) -> np.ndarray:
    """Tractrix horn contour (axially scaled to match the requested length).

    A tractrix curve with parameter `a = mouth_r` has a *natural* length set by
    `(throat_r, mouth_r)`. This helper computes that curve and then scales the
    axial coordinate to fit the requested `length`.

    Conventions
    ----------
    - `x=0` is the throat plane, `x=length` is the mouth plane.
    - Returns radius `r(x)` in meters.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if throat_r <= 0 or mouth_r <= 0:
        raise ValueError("throat_r and mouth_r must be > 0")
    if mouth_r <= throat_r:
        raise ValueError("mouth_r must be > throat_r for tractrix profile")

    a = float(mouth_r)
    x = np.asarray(x, dtype=float)

    # Distance from mouth to a given radius r on a tractrix curve.
    # x_mouth(r) = a * ln((a + sqrt(a^2 - r^2)) / r) - sqrt(a^2 - r^2)
    def _x_mouth(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        s = np.sqrt(np.maximum(a * a - r * r, 0.0))
        return a * np.log((a + s) / np.maximum(r, 1e-30)) - s

    # Natural length (mouth->throat) for the tractrix curve with parameter a=mouth_r.
    L0 = float(_x_mouth(np.array([float(throat_r)], dtype=float))[0])
    if not np.isfinite(L0) or L0 <= 0:
        return conical_profile(x, throat_r, mouth_r, length)

    # Build an interpolation table to invert x_mouth(r).
    n = int(max(512, n_table))
    r_grid = np.linspace(float(throat_r), float(mouth_r), n, dtype=float)
    x_grid = _x_mouth(r_grid)  # decreases from ~L0 to 0 as r increases
    # Ensure monotone and invert via interpolation (requires increasing abscissa).
    s_grid = x_grid[::-1]           # 0 .. L0
    r_rev = r_grid[::-1]            # mouth_r .. throat_r

    # Map requested axial coordinate to equivalent mouth-distance, with axial scaling.
    s = (float(length) - x) * (L0 / float(length))
    s = np.clip(s, 0.0, L0)
    return np.interp(s, s_grid, r_rev)


def tractrix_profile(x: np.ndarray, throat_r: float, mouth_r: float, length: float) -> np.ndarray:
    """Backward-compatible alias for the legacy "tractrix-like" easing curve.

    Use `sqrt_ease_profile` for the historical behavior, or `tractrix_horn_profile`
    for a tractrix-based horn contour.
    """
    warnings.warn(
        "`tractrix_profile` is a legacy 'tractrix-like' easing curve; "
        "use `sqrt_ease_profile` for the historical behavior or "
        "`tractrix_horn_profile` for a tractrix-based horn contour.",
        DeprecationWarning,
        stacklevel=2,
    )
    return sqrt_ease_profile(x, throat_r, mouth_r, length)


def hyperbolic_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Hyperbolic profile with adjustable sharpness parameter."""
    t = x / length
    cosh_max = np.cosh(sharpness)
    cosh_val = np.cosh(sharpness * t)
    return throat_r + (mouth_r - throat_r) * (cosh_val - 1) / (cosh_max - 1)


def oblate_spheroidal_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    *,
    opening_angle_deg: Optional[float] = None,
    opening_angle_rad: Optional[float] = None,
) -> np.ndarray:
    """Oblate-spheroidal-inspired waveguide profile (axisymmetric).

    This implements a common OSWG-style family where the wall approaches a cone of
    half-angle `opening_angle` (measured from the z-axis) but has curvature near
    the throat. The resulting contour is the radius of a hyperboloid-of-one-sheet
    (a constant-parameter surface in oblate spheroidal coordinates), axially scaled
    to match the requested `length`.

    Parameters
    ----------
    x : np.ndarray
        Axial coordinates from throat (0) to mouth (L).
    throat_r, mouth_r : float
        Throat and mouth radii.
    length : float
        Axial length L.
    opening_angle_deg/opening_angle_rad : float, optional
        Target asymptotic half-angle from z-axis (often chosen to approximate
        coverage angle). If not provided, defaults to the simple conical angle
        implied by (throat_r, mouth_r, length).

    Notes
    -----
    This profile is meant as a practical OS-style option. For designs that must
    match a compression-driver exit cone and add baffle tangency, prefer `cts_profile`.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if throat_r <= 0 or mouth_r <= 0:
        raise ValueError("throat_r and mouth_r must be > 0")
    if mouth_r <= throat_r:
        raise ValueError("mouth_r must be > throat_r for oblate spheroidal profile")
    if opening_angle_deg is not None and opening_angle_rad is not None:
        raise TypeError("Pass only one of `opening_angle_deg` or `opening_angle_rad`.")

    alpha_min = float(np.arctan((float(mouth_r) - float(throat_r)) / float(length)))
    if opening_angle_deg is not None:
        alpha = float(np.deg2rad(float(opening_angle_deg)))
        if alpha + 1e-12 < alpha_min:
            raise ValueError(
                "opening_angle is too small to reach the requested mouth radius in the given length; "
                f"need >= {np.rad2deg(alpha_min):.3f}° for this geometry."
            )
    elif opening_angle_rad is not None:
        alpha = float(opening_angle_rad)
        if alpha + 1e-12 < alpha_min:
            raise ValueError(
                "opening_angle is too small to reach the requested mouth radius in the given length; "
                f"need >= {alpha_min:.6g} rad for this geometry."
            )
    else:
        # Minimum feasible (degenerates to conical for this parameterization).
        alpha = alpha_min

    # Avoid degenerate/vertical cases.
    alpha = float(np.clip(alpha, 1e-6, np.deg2rad(89.9)))
    tan_a = float(np.tan(alpha))
    tan2 = float(tan_a * tan_a)

    # Solve for the axial offset z0 that makes the hyperboloid segment hit both endpoints:
    # r(0)^2 = w^2 + (z0 tan α)^2
    # r(L)^2 = w^2 + ((L+z0) tan α)^2
    # => z0 = (r1^2 - r0^2)/(2 L tan^2 α) - L/2, and w^2 = r0^2 - (z0 tan α)^2.
    r0 = float(throat_r)
    r1 = float(mouth_r)
    L = float(length)
    z0 = (r1 * r1 - r0 * r0) / (2.0 * L * tan2) - 0.5 * L
    w2 = r0 * r0 - (z0 * tan_a) ** 2
    if not np.isfinite(w2):
        raise ValueError("oblate_spheroidal_profile produced non-finite parameters")
    # Numerical tolerance: the degenerate limit (conical) can land very close to 0.
    eps = max(1e-24, 1e-12 * r0 * r0)
    if w2 < -eps:
        raise ValueError(
            "oblate_spheroidal_profile geometry is infeasible for the requested opening_angle; "
            "choose a larger angle or adjust throat/mouth/length."
        )
    if w2 <= eps:
        # Degenerate case: this parameterization collapses to a cone.
        return conical_profile(x, throat_r, mouth_r, length)

    x = np.asarray(x, dtype=float)
    z = x + float(z0)
    return np.sqrt(float(w2) + (z * tan_a) ** 2)


def os_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    *,
    opening_angle_deg: Optional[float] = None,
    opening_angle_rad: Optional[float] = None,
) -> np.ndarray:
    """Alias for `oblate_spheroidal_profile`."""
    return oblate_spheroidal_profile(
        x,
        throat_r,
        mouth_r,
        length,
        opening_angle_deg=opening_angle_deg,
        opening_angle_rad=opening_angle_rad,
    )


def compute_exit_cone_angle_deg(
    throat_diameter_m: float,
    mouth_diameter_m: float,
    length_m: float,
) -> float:
    """
    Compute the half-angle of a conical exit cone from its geometry.

    This is useful for matching a waveguide throat to a compression driver's
    exit cone angle.

    Parameters
    ----------
    throat_diameter_m : float
        Throat (smaller) diameter in meters.
    mouth_diameter_m : float
        Mouth (larger) diameter in meters.
    length_m : float
        Axial length of the cone in meters.

    Returns
    -------
    float
        Half-angle in degrees.
    """
    delta_r = (mouth_diameter_m - throat_diameter_m) / 2
    return float(np.rad2deg(np.arctan(delta_r / length_m)))


def cts_profile(
    x: np.ndarray,
    throat_r: float,
    mouth_r: float,
    length: float,
    *,
    throat_blend: float = 0.0,
    transition: Optional[float] = 0.75,
    driver_exit_angle_deg: Optional[float] = None,
    driver_exit_angle_rad: Optional[float] = None,
    throat_angle_deg: Optional[float] = None,
    throat_angle_rad: Optional[float] = None,
    tangency: float = 1.0,
    mouth_roll: float = 0.0,
    curvature_regularizer: float = 1.0,
    mid_curvature: float = 0.0,
) -> np.ndarray:
    """
    CTS (Conical-to-Tangent Smoothstep) profile for "constant directivity" style horns.

    The wall angle is measured from the z-axis (axis of symmetry). The baffle is the
    XY plane at z=0, and the waveguide extends into -z. "Tangent to baffle" means the
    wall approaches z=0 horizontally (wall angle = 90 deg from z-axis).

    Design goals:
    - Match driver exit cone angle at the throat for smooth acoustic coupling
    - Conical-like propagation for most of the length (helps maintain beamwidth above Fc)
    - Smooth transition of wall angle near the mouth toward horizontal (baffle-tangent)
    - Adjustable mouth tangency for tuning the baffle transition

    The profile has 3 sections:
    1. Throat blend (0 → throat_blend): driver_exit_angle → conical_angle
    2. Conical (throat_blend → transition): constant conical_angle
    3. Mouth blend (transition → 1): conical_angle → mouth_angle

    Parameters
    ----------
    x : np.ndarray
        Axial coordinates from throat (0) to mouth (L).
    throat_r, mouth_r : float
        Throat and mouth radii.
    length : float
        Axial length L.
    throat_blend : float, optional
        Normalized location t_tb in [0, transition) where the throat blend ends.
        - 0.0 (default): No throat blend, starts directly with conical angle
        - 0.1-0.2: Typical values for blending from driver exit angle
        Must be less than `transition`.
    transition : float, optional
        Normalized location t_c in [0, 1) where the conical section ends and
        the mouth blend begins. Default 0.75.
    driver_exit_angle_deg, driver_exit_angle_rad : float, optional
        Driver exit cone half-angle. Used for throat blend section.
        If not provided and throat_blend > 0, defaults to conical_angle.
    throat_angle_deg, throat_angle_rad : float, optional
        Override for the conical section angle. If not provided, auto-computed
        to satisfy r(L) = mouth_r.
    tangency : float, optional
        Controls how tangent the mouth is to the baffle (XY plane at z=0). Range [0, 1]:
        - 1.0 (default): Maximum achievable tangency (geometry-dependent)
        - 0.5: Half-tangent
        - 0.0: No tangent smoothing (conical exit)

    Returns
    -------
    np.ndarray
        Radius r(x) along the profile.

    Notes
    -----
    Coordinate system:
    - Baffle is XY plane at z=0
    - Mouth is at z=0, throat is at z=-L
    - Wall angle alpha is measured from the z-axis
    - alpha=0 means vertical wall, alpha=90 deg means horizontal (tangent to baffle)

    The profile is constructed by numerical integration of tan(alpha(t)) where
    alpha varies smoothly according to the 3-section definition.

    For tangency=1.0, the mouth wall angle approaches horizontal (limited by
    the geometric constraint that r(L) = mouth_r).
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if mouth_r <= throat_r:
        raise ValueError("mouth_r must be > throat_r for CTS profile")
    if not (0.0 <= tangency <= 1.0):
        raise ValueError("tangency must be in [0, 1]")
    if not (0.0 <= float(mouth_roll) < 1.0):
        raise ValueError("mouth_roll must be in [0, 1)")
    if float(curvature_regularizer) < 0.0:
        raise ValueError("curvature_regularizer must be >= 0")
    if not (0.0 <= float(mid_curvature) <= 1.0):
        raise ValueError("mid_curvature must be in [0, 1]")
    if not (0.0 <= throat_blend < 1.0):
        raise ValueError("throat_blend must be in [0, 1)")

    # Validate angle parameter combinations
    if throat_angle_deg is not None and throat_angle_rad is not None:
        raise TypeError("Pass only one of `throat_angle_deg` or `throat_angle_rad`.")
    if driver_exit_angle_deg is not None and driver_exit_angle_rad is not None:
        raise TypeError("Pass only one of `driver_exit_angle_deg` or `driver_exit_angle_rad`.")

    delta_r = float(mouth_r - throat_r)
    x = np.asarray(x, dtype=float)
    t = x / float(length)
    tau = float(tangency)
    t_tb = float(throat_blend)
    roll = float(mouth_roll)
    reg = float(curvature_regularizer)
    midk = float(mid_curvature)

    # Set up transition point
    if transition is None:
        transition = 0.75
    t_c = float(transition)
    if not (0.0 <= t_c < 1.0):
        raise ValueError("transition must be in [0, 1)")
    if t_tb >= t_c:
        raise ValueError(f"throat_blend ({t_tb}) must be < transition ({t_c})")

    def _quintic_smoothstep(u: np.ndarray) -> np.ndarray:
        """Quintic smoothstep S(u) = 10u^3 - 15u^4 + 6u^5, with S(0)=0, S(1)=1."""
        u = np.clip(u, 0.0, 1.0)
        return 10 * u**3 - 15 * u**4 + 6 * u**5

    def _mouth_roll_map(u: np.ndarray) -> np.ndarray:
        """
        Map u∈[0,1] to a "rolled" coordinate that reaches 1 earlier.

        `mouth_roll=0` -> identity
        `mouth_roll→1` -> aggressive ease-out (spends more of the blend near α_mouth)
        """
        if roll <= 0.0:
            return np.clip(u, 0.0, 1.0)
        gamma = 1.0 / max(1e-9, (1.0 - roll))
        u = np.clip(u, 0.0, 1.0)
        return 1.0 - np.power((1.0 - u), gamma)

    def _adaptive_min_conical_factor() -> float:
        """
        Adaptive lower bound for the mid-section conical angle.

        This replaces the older fixed `0.70` and scales with:
        - expansion ratio (mouth/throat)
        - available mouth-blend length (1 - t_c)
        """
        expansion = float(mouth_r) / float(throat_r)
        mouth_blend = float(max(0.0, 1.0 - t_c))
        # More mouth blend and higher expansion allow more "room" for tangency,
        # but clamp to keep the mid-section meaningfully conical.
        # Heuristic scaling in a conservative range.
        loge = float(np.log2(max(expansion, 1.0 + 1e-12)))
        f = 0.90 - 0.25 * mouth_blend - 0.05 * loge
        return float(np.clip(f, 0.60, 0.92))

    def _get_driver_exit_angle_rad() -> Optional[float]:
        """Get driver exit angle in radians, if specified."""
        if driver_exit_angle_deg is not None:
            return float(np.deg2rad(float(driver_exit_angle_deg)))
        if driver_exit_angle_rad is not None:
            return float(driver_exit_angle_rad)
        return None

    def _get_throat_angle_rad() -> Optional[float]:
        """Get throat (conical) angle in radians, if specified."""
        if throat_angle_deg is not None:
            return float(np.deg2rad(float(throat_angle_deg)))
        if throat_angle_rad is not None:
            return float(throat_angle_rad)
        return None

    def _compute_r_from_angles(
        alpha_driver: float,
        alpha_conical: float,
        alpha_mouth: float,
        n_fine: int = 1000,
    ) -> float:
        """
        Compute mouth radius for given 3-section angle configuration.

        Integrates dr/dx = tan(alpha(t)) where alpha varies across 3 sections:
        1. Throat blend (0 → t_tb): alpha_driver → alpha_conical
        2. Conical (t_tb → t_c): constant alpha_conical
        3. Mouth blend (t_c → 1): alpha_conical → alpha_mouth
        """
        t_fine = np.linspace(0.0, 1.0, n_fine)
        alpha = np.empty(n_fine)

        for i, ti in enumerate(t_fine):
            if ti < t_tb and t_tb > 0:
                # Section 1: throat blend
                u = ti / t_tb
                S = 10 * u**3 - 15 * u**4 + 6 * u**5
                alpha[i] = alpha_driver + (alpha_conical - alpha_driver) * S
            elif ti < t_c:
                # Section 2: conical (constant angle)
                if midk > 0.0 and t_c > t_tb:
                    # Add a gentle zero-endpoint bump to mimic OS-like curvature
                    # while preserving conical endpoints.
                    u = (ti - t_tb) / (t_c - t_tb)
                    bump = np.sin(np.pi * np.clip(u, 0.0, 1.0))
                    alpha[i] = alpha_conical + (0.15 * midk * alpha_conical) * bump
                else:
                    alpha[i] = alpha_conical
            else:
                # Section 3: mouth blend
                if t_c < 1.0:
                    u = (ti - t_c) / (1.0 - t_c)
                    u = _mouth_roll_map(u)
                    S = 10 * u**3 - 15 * u**4 + 6 * u**5
                    alpha[i] = alpha_conical + (alpha_mouth - alpha_conical) * S
                else:
                    alpha[i] = alpha_conical

        tan_alpha = np.tan(alpha)
        dt = t_fine[1] - t_fine[0]
        # NumPy exposes trapezoidal integration as `trapz` (NumPy < 2.0 has no
        # `trapezoid` alias).
        integral = np.trapz(tan_alpha, dx=dt)

        return float(throat_r) + float(length) * integral

    def _find_mouth_angle_for_conical(
        alpha_driver: float,
        alpha_conical: float,
    ) -> float:
        """Find the mouth angle that gives r(L) = mouth_r for a fixed conical angle.

        This keeps the conical section at a sensible angle and adjusts the mouth
        blend to hit the target radius.
        """
        # Binary search on mouth angle
        lo = alpha_conical  # Mouth angle must be >= conical angle
        hi = np.deg2rad(89.9)

        for _ in range(60):
            mid = (lo + hi) / 2
            r = _compute_r_from_angles(alpha_driver, alpha_conical, mid)
            if abs(r - mouth_r) < 1e-10 * mouth_r:
                return mid
            if r > mouth_r:
                hi = mid  # Mouth angle too large, reduce it
            else:
                lo = mid  # Mouth angle too small, increase it

        return mid

    def _find_max_mouth_angle_with_conical(alpha_driver: float, alpha_conical: float) -> float:
        """Find maximum achievable mouth angle for the geometry with fixed conical angle."""
        # The max mouth angle is limited by the geometry constraint r(L) = mouth_r
        # With fixed conical angle, we find what mouth angle achieves this
        return _find_mouth_angle_for_conical(alpha_driver, alpha_conical)

    def _compute_profile_3section(
        t_query: np.ndarray,
        alpha_driver: float,
        alpha_conical: float,
        alpha_mouth: float,
    ) -> np.ndarray:
        """Compute r(t) at requested `t_query` using 3-section integration."""
        # Build a fine reference grid
        n_fine = 1000
        t_fine = np.linspace(0.0, 1.0, n_fine)
        alpha_fine = np.empty(n_fine)

        for i, ti in enumerate(t_fine):
            if ti < t_tb and t_tb > 0:
                # Section 1: throat blend
                u = ti / t_tb
                S = 10 * u**3 - 15 * u**4 + 6 * u**5
                alpha_fine[i] = alpha_driver + (alpha_conical - alpha_driver) * S
            elif ti < t_c:
                # Section 2: conical
                if midk > 0.0 and t_c > t_tb:
                    u = (ti - t_tb) / (t_c - t_tb)
                    bump = np.sin(np.pi * np.clip(u, 0.0, 1.0))
                    alpha_fine[i] = alpha_conical + (0.15 * midk * alpha_conical) * bump
                else:
                    alpha_fine[i] = alpha_conical
            else:
                # Section 3: mouth blend
                if t_c < 1.0:
                    u = (ti - t_c) / (1.0 - t_c)
                    u = _mouth_roll_map(u)
                    S = 10 * u**3 - 15 * u**4 + 6 * u**5
                    alpha_fine[i] = alpha_conical + (alpha_mouth - alpha_conical) * S
                else:
                    alpha_fine[i] = alpha_conical

        tan_alpha_fine = np.tan(alpha_fine)

        # Cumulative trapezoidal integration
        dt = t_fine[1] - t_fine[0]
        cumsum = np.zeros(n_fine)
        cumsum[1:] = np.cumsum(0.5 * (tan_alpha_fine[:-1] + tan_alpha_fine[1:]) * dt)
        r_fine = float(throat_r) + float(length) * cumsum

        # Interpolate to requested t values
        return np.interp(np.asarray(t_query, dtype=float), t_fine, r_fine)

    # Compute baseline conical angle (simple geometry)
    simple_conical_angle = np.arctan(delta_r / float(length))

    # Determine driver exit angle
    alpha_driver_rad = _get_driver_exit_angle_rad()
    if alpha_driver_rad is None:
        # Default: use conical angle (no throat blend effect)
        alpha_driver_rad = simple_conical_angle

    # Determine conical section angle
    alpha_conical_specified = _get_throat_angle_rad()

    if alpha_conical_specified is not None:
        # User specified conical angle - use it directly
        alpha_conical_rad = alpha_conical_specified
        alpha_mouth_rad = _find_mouth_angle_for_conical(alpha_driver_rad, alpha_conical_rad)
    else:
        if tau == 0.0:
            alpha_conical_rad = simple_conical_angle
            alpha_mouth_rad = simple_conical_angle
        else:
            # Adaptive conical reduction budget (replaces fixed 0.70).
            min_conical_factor = _adaptive_min_conical_factor()
            alpha_c_min = max(1e-8, simple_conical_angle * float(min_conical_factor))
            alpha_c_max = simple_conical_angle

            # A "max tangency" reference for this geometry (given our conical floor).
            alpha_m_at_floor = _find_mouth_angle_for_conical(alpha_driver_rad, alpha_c_min)

            # Target mouth angle based on tangency knob.
            alpha_m_target = simple_conical_angle + tau * (alpha_m_at_floor - simple_conical_angle)

            # Curvature-regularized 1D search over alpha_conical.
            # For each candidate alpha_conical we solve alpha_mouth so r(L)=mouth_r,
            # then score by peak curvature in the mouth blend + a soft penalty for
            # deviating from alpha_m_target.
            n_scan = 21
            alpha_cs = np.linspace(alpha_c_min, alpha_c_max, n_scan)
            best = None
            best_alpha_c = None
            best_alpha_m = None

            # Normalization for mouth-angle penalty (avoid over-weighting in small ranges).
            denom = max(1e-6, float(abs(alpha_m_at_floor - simple_conical_angle)))

            for alpha_c in alpha_cs:
                alpha_m = _find_mouth_angle_for_conical(alpha_driver_rad, float(alpha_c))
                # Evaluate curvature on a modest grid independent of the caller's sampling.
                t_eval = np.linspace(0.0, 1.0, 301, dtype=float)
                x_eval = float(length) * t_eval
                r_tmp = _compute_profile_3section(t_eval, alpha_driver_rad, float(alpha_c), float(alpha_m))
                # Peak curvature in mouth blend region (t >= t_c).
                # Use existing curvature helper (central differences).
                curv = compute_profile_curvature(x_eval, r_tmp)
                mask = t_eval >= float(t_c) - 1e-12
                kmax = float(np.max(curv[mask])) if np.any(mask) else float(np.max(curv))
                curv_term = float(np.log1p(kmax * float(length)))
                ang_term = float(((alpha_m - alpha_m_target) / denom) ** 2)
                cost = curv_term + float(reg) * float(2.0 * tau) * ang_term
                if best is None or cost < best:
                    best = cost
                    best_alpha_c = float(alpha_c)
                    best_alpha_m = float(alpha_m)

            alpha_conical_rad = float(best_alpha_c)
            alpha_mouth_rad = float(best_alpha_m)

    # Compute final profile
    r = _compute_profile_3section(t, alpha_driver_rad, alpha_conical_rad, alpha_mouth_rad)

    return r


def cts_mouth_angle_deg(
    throat_r: float,
    mouth_r: float,
    length: float,
    throat_blend: float = 0.0,
    transition: float = 0.75,
    driver_exit_angle_deg: Optional[float] = None,
    tangency: float = 1.0,
    mouth_roll: float = 0.0,
    curvature_regularizer: float = 1.0,
    mid_curvature: float = 0.0,
) -> float:
    """
    Compute the mouth wall angle (in degrees) for a CTS profile.

    The wall angle is measured from the z-axis (axis of symmetry):
    - 0 deg = vertical wall (parallel to z-axis)
    - 90 deg = horizontal wall (tangent to baffle at z=0)

    This function computes the profile using cts_profile and measures the
    actual wall angle at the mouth from the derivative.

    Parameters
    ----------
    throat_r, mouth_r : float
        Throat and mouth radii.
    length : float
        Axial length L.
    throat_blend : float
        Normalized throat blend point (0 = no throat blend).
    transition : float
        Normalized transition point t_c.
    driver_exit_angle_deg : float, optional
        Driver exit angle for throat blend section.
    tangency : float
        Tangency factor in [0, 1].

    Returns
    -------
    float
        Mouth wall angle in degrees.
        - For tangency=1.0, returns the maximum achievable angle for this geometry
        - For tangency=0.0, returns the conical angle
    """
    # Compute profile with fine resolution
    x = np.linspace(0, length, 500)
    r = cts_profile(
        x,
        throat_r,
        mouth_r,
        length,
        throat_blend=throat_blend,
        transition=transition,
        driver_exit_angle_deg=driver_exit_angle_deg,
        tangency=tangency,
        mouth_roll=mouth_roll,
        curvature_regularizer=curvature_regularizer,
        mid_curvature=mid_curvature,
    )

    # Compute wall angle at mouth from the derivative
    dr_dx = (r[-1] - r[-2]) / (x[-1] - x[-2])
    mouth_angle_rad = np.arctan(dr_dx)

    return float(np.rad2deg(mouth_angle_rad))


# =============================================================================
# Profile Curvature Utilities
# =============================================================================


def compute_profile_curvature(
    x: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Compute curvature of a 2D profile r(x).

    For a curve y = r(x), the curvature is:
        κ = |r''| / (1 + r'^2)^(3/2)

    Parameters
    ----------
    x : np.ndarray
        Axial positions (assumed uniformly spaced for accurate derivatives).
    r : np.ndarray
        Radius values at each x position.

    Returns
    -------
    np.ndarray
        Curvature values κ at each point. Same length as input.
        Units: 1/m (inverse of radius of curvature).

    Notes
    -----
    - High curvature = sharp bend (small radius of curvature)
    - Zero curvature = straight line (infinite radius of curvature)
    - Uses central differences for interior points, one-sided at boundaries.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)

    if len(x) < 3:
        return np.zeros_like(x)

    # First derivative (central difference, one-sided at boundaries)
    dr = np.gradient(r, x)

    # Second derivative
    d2r = np.gradient(dr, x)

    # Curvature: κ = |d²r/dx²| / (1 + (dr/dx)²)^(3/2)
    denominator = (1.0 + dr**2) ** 1.5
    curvature = np.abs(d2r) / np.maximum(denominator, 1e-12)

    return curvature


def compute_curvature_radius(
    x: np.ndarray,
    r: np.ndarray,
    min_radius: float = 1e-6,
) -> np.ndarray:
    """Compute radius of curvature along a profile.

    Parameters
    ----------
    x : np.ndarray
        Axial positions.
    r : np.ndarray
        Radius values at each x position.
    min_radius : float, optional
        Minimum radius to return (avoids infinity for straight sections).
        Default 1e-6 m.

    Returns
    -------
    np.ndarray
        Radius of curvature (1/κ) at each point, in meters.
        Capped at 1e6 m for nearly-straight sections.

    Notes
    -----
    - Small radius of curvature = sharp bend (needs finer mesh)
    - Large radius of curvature = gentle curve or straight (can use coarser mesh)
    """
    curvature = compute_profile_curvature(x, r)
    radius = np.where(curvature > 1e-12, 1.0 / curvature, 1e6)
    return np.maximum(radius, min_radius)


def check_min_curvature_radius(
    x: np.ndarray,
    r: np.ndarray,
    min_radius_m: float,
) -> tuple[bool, float, float]:
    """Check if profile meets minimum curvature radius constraint.

    This is a manufacturability constraint: very sharp bends are hard to
    machine/mold and can cause flow separation at high SPL.

    Parameters
    ----------
    x : np.ndarray
        Axial positions.
    r : np.ndarray
        Radius values at each x position.
    min_radius_m : float
        Minimum acceptable radius of curvature in meters.

    Returns
    -------
    ok : bool
        True if all curvature radii >= min_radius_m.
    actual_min : float
        Actual minimum radius of curvature found (m).
    location : float
        x-position where minimum radius occurs.

    Examples
    --------
    >>> x = np.linspace(0, 0.1, 100)
    >>> r = exponential_profile(x, 0.01, 0.05, 0.1)
    >>> ok, actual_min, loc = check_min_curvature_radius(x, r, 0.005)
    >>> if not ok:
    ...     print(f"Curvature radius {actual_min*1000:.1f}mm < {0.005*1000:.1f}mm at x={loc*1000:.1f}mm")
    """
    radii = compute_curvature_radius(x, r)
    idx_min = np.argmin(radii)
    actual_min = float(radii[idx_min])
    location = float(x[idx_min])
    ok = actual_min >= min_radius_m

    return ok, actual_min, location


def curvature_based_element_size(
    x: np.ndarray,
    r: np.ndarray,
    h_min: float,
    h_max: float,
    curvature_factor: float = 1.0,
) -> np.ndarray:
    """Compute element sizes based on local profile curvature.

    Regions with high curvature get smaller elements (h_min),
    regions with low curvature get larger elements (h_max).

    Parameters
    ----------
    x : np.ndarray
        Axial positions.
    r : np.ndarray
        Radius values at each x position.
    h_min : float
        Minimum element size (at highest curvature).
    h_max : float
        Maximum element size (at zero curvature).
    curvature_factor : float, optional
        Scaling factor for curvature sensitivity. Higher = more aggressive
        refinement at curved regions. Default 1.0.

    Returns
    -------
    np.ndarray
        Recommended element size at each x position.

    Notes
    -----
    The sizing function is:
        h(x) = h_max - (h_max - h_min) * tanh(curvature_factor * κ * R_ref)

    where R_ref is a reference scale (mouth radius) to make the
    curvature-scaling dimensionless.
    """
    curvature = compute_profile_curvature(x, r)

    # Reference scale: use maximum radius for dimensionless scaling
    r_ref = max(float(np.max(r)), 0.01)

    # Normalized curvature (dimensionless)
    kappa_norm = curvature * r_ref * curvature_factor

    # Map to element size via tanh (saturates for very high curvature)
    weight = np.tanh(kappa_norm)
    h = h_max - (h_max - h_min) * weight

    return h
