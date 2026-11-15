"""Display: brightness stretch, the animated Plotly movie (accurate axes in arcsec) and GIF export."""

from __future__ import annotations

import io
import warnings

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

from .align import offset_to_pix, pix_to_offset  # noqa: F401  (re-exported for the UI)


def stretch(
    cube: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.7,
    a: float = 0.1,
    per_frame: bool = False,
) -> np.ndarray:
    """asinh stretch to [0, 1]. Default: one global stretch so real brightness changes stay visible."""

    def one(x, lo, hi):
        z = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
        return np.arcsinh(z / a) / np.arcsinh(1 / a)

    out = np.empty_like(cube, dtype=np.float32)
    if per_frame:
        for i in range(len(cube)):
            f = cube[i][np.isfinite(cube[i])]
            lo, hi = np.percentile(f, [lo_pct, hi_pct]) if f.size else (0, 1)
            out[i] = one(cube[i], lo, hi)
    else:
        f = cube[np.isfinite(cube)]
        lo, hi = np.percentile(f, [lo_pct, hi_pct]) if f.size else (0, 1)
        out[:] = one(cube, lo, hi)
    return out


def limits(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> tuple[float, float]:
    """Robust display range of an image or cube (percentiles of the finite pixels)."""
    good = values[np.isfinite(values)]
    if good.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(good, [lo_pct, hi_pct])
    return float(lo), float(hi)


def _nanmean(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels at the edges are expected
        return np.nanmean(stack, axis=0)


def windowed(cube: np.ndarray, group: list[int], k: int, window: int = 1) -> np.ndarray:
    """Frame k averaged with its neighbours in time (`window` frames in all, centred on k).

    `group` lists the frames that may be combined with k (the ones from the same detector), in
    time order. Averaging neighbours trades time resolution for a cleaner picture.
    """
    if window <= 1 or k not in group:
        return cube[k]
    at, half = group.index(k), window // 2
    return _nanmean(cube[group[max(at - half, 0) : at + half + 1]])


def static_sky(cube: np.ndarray, group: list[int]) -> np.ndarray:
    """What does not change: the per-pixel median of a group of frames."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(cube[group], axis=0)


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.flipud(rgba), "RGBA").save(buf, format="PNG")  # flip: North up
    return buf.getvalue()


def gray_rgba(a: np.ndarray, lo: float, hi: float, soft: float = 0.1) -> np.ndarray:
    z = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    g = (np.nan_to_num(np.arcsinh(z / soft) / np.arcsinh(1 / soft)) * 255).astype(np.uint8)
    return np.dstack([g, g, g, np.where(np.isfinite(a), 255, 0).astype(np.uint8)])


def diverging_rgba(a: np.ndarray, lim: float) -> np.ndarray:
    """Signed difference image: red = brighter later (arrived / brightened), blue = brighter earlier (left / faded)."""
    z = np.clip(np.nan_to_num(a) / max(lim, 1e-12), -1, 1)[..., None]
    base, pos, neg = np.array([18, 22, 38.0]), np.array([255, 90, 70.0]), np.array([80, 150, 255.0])
    rgb = base + (pos - base) * np.clip(z, 0, 1) + (neg - base) * np.clip(-z, 0, 1)
    return np.dstack([rgb.astype(np.uint8), np.where(np.isfinite(a), 255, 0).astype(np.uint8)])


def movie_figure(
    cube01: np.ndarray,
    labels: list[str],
    scale_arcsec: float,
    pred_offsets=None,
    meas_offsets=None,
    frame_ms: int = 350,
    title: str = "",
) -> go.Figure:
    """cube01: (N, ny, nx) in [0,1] with East left / North up. pred/meas_offsets: (N, 2) arrays of (east, north)
    arcsec or None; NaN rows are skipped. A faint line shows the predicted path."""
    n, ny, nx = cube01.shape
    east = ((nx - 1) / 2.0 - np.arange(nx)) * scale_arcsec  # decreasing with column index
    north = (np.arange(ny) - (ny - 1) / 2.0) * scale_arcsec
    xs, z_all = (
        east[::-1],
        np.round(cube01[:, :, ::-1], 3),
    )  # ascending x; axis reversed below -> East on left

    def heat(k):
        return go.Heatmap(
            z=z_all[k],
            x=xs,
            y=north,
            colorscale="Gray",
            showscale=False,
            zmin=0,
            zmax=1,
            hovertemplate="ΔRA %{x:.1f}″  ΔDec %{y:.1f}″<extra></extra>",
        )

    def pts(arr, k, **kw):
        if arr is None or not np.all(np.isfinite(arr[k])):
            return go.Scatter(x=[], y=[], **kw)
        return go.Scatter(x=[arr[k][0]], y=[arr[k][1]], **kw)

    pred_kw = dict(
        mode="markers",
        name="Predicted position",
        marker=dict(symbol="circle-open", size=26, color="#ffcc00", line=dict(width=2)),
    )
    meas_kw = dict(
        mode="markers",
        name="Measured centroid",
        marker=dict(symbol="x-thin-open", size=14, color="#ff4d4d", line=dict(width=2)),
    )
    data = [heat(0), pts(pred_offsets, 0, **pred_kw), pts(meas_offsets, 0, **meas_kw)]
    if pred_offsets is not None:
        ok = np.all(np.isfinite(pred_offsets), axis=1)
        data.append(
            go.Scatter(
                x=pred_offsets[ok, 0],
                y=pred_offsets[ok, 1],
                mode="lines",
                name="Predicted path",
                line=dict(color="rgba(255,204,0,0.35)", width=1, dash="dot"),
                hoverinfo="skip",
            )
        )
    frames = [
        go.Frame(
            data=[heat(k), pts(pred_offsets, k, **pred_kw), pts(meas_offsets, k, **meas_kw)],
            traces=[0, 1, 2],
            name=str(k),
        )
        for k in range(n)
    ]
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=640,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(title="ΔRA·cosδ (arcsec) ← East", autorange="reversed", constrain="domain"),
        yaxis=dict(
            title="ΔDec (arcsec) ↑ North", scaleanchor="x", scaleratio=1, constrain="domain"
        ),
        legend=dict(orientation="h", y=1.04, x=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.12,
                len=0.88,
                currentvalue=dict(prefix="Observed: "),
                pad=dict(t=30),
                steps=[
                    dict(
                        method="animate",
                        label=lab,
                        args=[
                            [str(k)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for k, lab in enumerate(labels)
                ],
            )
        ],
    )
    return fig


def gif_bytes(
    cube01: np.ndarray, labels: list[str], fps: int = 6, zoom: int = 8, watermark: str = ""
) -> bytes:
    """Shareable GIF (North up, East left, same orientation as the interactive movie)."""
    frames = []
    for k in range(len(cube01)):
        a = np.nan_to_num(cube01[k], nan=0.0)
        im = (
            Image.fromarray((np.flipud(a) * 255).astype(np.uint8))
            .resize((a.shape[1] * zoom, a.shape[0] * zoom), Image.NEAREST)
            .convert("RGB")
        )
        d = ImageDraw.Draw(im)
        d.text((6, 6), labels[k], fill=(255, 230, 120))
        if watermark:
            d.text((6, im.height - 14), watermark, fill=(255, 120, 120))
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
    )
    return buf.getvalue()
