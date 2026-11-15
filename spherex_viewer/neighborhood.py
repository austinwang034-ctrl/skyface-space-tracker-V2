"""Gaia reference stars: cone search (cached), epoch propagation, motion statistics and the neighbourhood map."""

from __future__ import annotations

import hashlib

import astropy.units as u
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from .config import CATALOG_DIR

GAIA_REF_EPOCH = 2016.0


def gaia_cone(center: SkyCoord, radius_arcmin: float, mag_limit: float = 18.0) -> Table:
    """Gaia DR3 sources in a cone (needs network the first time, then served from the disk cache)."""
    key = f"{center.ra.deg:.5f}_{center.dec.deg:.5f}_{radius_arcmin:.2f}_{mag_limit:.1f}"
    path = CATALOG_DIR / f"gaia_{hashlib.sha1(key.encode()).hexdigest()[:16]}.ecsv"
    if path.exists():
        return Table.read(path)
    errors = []
    tbl = None
    for name, fn in (("ESA Gaia archive", _gaia_esa), ("MAST Gaia mirror", _gaia_mast)):
        try:
            tbl = fn(center, radius_arcmin, mag_limit)
            tbl.meta["source"] = name
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    if tbl is None:
        raise RuntimeError(" | ".join(errors))
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    tbl.write(path, format="ascii.ecsv", overwrite=True)
    return tbl


GAIA_COLS = ("source_id", "ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "ref_epoch")


def _gaia_esa(center: SkyCoord, radius_arcmin: float, mag_limit: float) -> Table:
    from astroquery.gaia import Gaia

    q = (
        f"SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, ref_epoch "
        f"FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {center.ra.deg:.8f}, {center.dec.deg:.8f}, {radius_arcmin / 60.0:.6f})) "
        f"AND phot_g_mean_mag < {mag_limit}"
    )
    return Gaia.launch_job_async(q).get_results()


def normalise_gaia(raw: Table, mag_limit: float) -> Table:
    """Reduce any Gaia DR3 table (whatever the column capitalisation) to the columns this package uses."""
    low = {c.lower(): c for c in raw.colnames}
    missing = [c for c in GAIA_COLS[:7] if c not in low]
    if missing:
        raise ValueError(f"Gaia table lacks columns {missing}")
    out = Table(
        {
            c: (
                np.ma.filled(np.ma.asarray(raw[low[c]], dtype=float), np.nan)
                if c != "source_id"
                else [str(v) for v in raw[low[c]]]
            )
            for c in GAIA_COLS[:7]
        }
    )
    out["ref_epoch"] = (
        np.ma.filled(np.ma.asarray(raw[low["ref_epoch"]], dtype=float), GAIA_REF_EPOCH)
        if "ref_epoch" in low
        else np.full(len(out), GAIA_REF_EPOCH)
    )
    return out[np.asarray(out["phot_g_mean_mag"]) < mag_limit]


def _gaia_mast(center: SkyCoord, radius_arcmin: float, mag_limit: float) -> Table:
    """Gaia DR3 from the copy hosted at MAST (STScI), through its VO cone-search service."""
    from io import BytesIO

    import requests

    from .config import MAST_GAIA_CONE

    r = requests.get(
        MAST_GAIA_CONE,
        params={
            "format": "votable",
            "RA": f"{center.ra.deg:.8f}",
            "DEC": f"{center.dec.deg:.8f}",
            "SR": f"{radius_arcmin / 60.0:.6f}",
        },
        timeout=60,
    )
    r.raise_for_status()
    return normalise_gaia(Table.read(BytesIO(r.content), format="votable"), mag_limit)


def propagate(tbl: Table, when: Time) -> SkyCoord:
    """Positions of all rows at `when` using each star's proper motion (missing PM -> 0)."""
    pmra = np.nan_to_num(np.asarray(tbl["pmra"], dtype=float))
    pmdec = np.nan_to_num(np.asarray(tbl["pmdec"], dtype=float))
    c = SkyCoord(
        ra=np.asarray(tbl["ra"], dtype=float) * u.deg,
        dec=np.asarray(tbl["dec"], dtype=float) * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(GAIA_REF_EPOCH, format="jyear"),
        frame="icrs",
    )
    moved = c.apply_space_motion(new_obstime=when)
    return SkyCoord(
        moved.ra, moved.dec, frame="icrs"
    )  # plain ICRS: drop obstime so it mixes with other coords


def motion_table(tbl: Table) -> pd.DataFrame:
    """Per-star total proper motion (mas/yr) and tangential velocity (km/s) where a parallax exists."""
    df = tbl.to_pandas()
    for c in ("pmra", "pmdec", "parallax"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pm_total_masyr"] = np.hypot(df.pmra.fillna(0), df.pmdec.fillna(0))
    good = df.parallax > 0.5
    df["vtan_kms"] = np.where(
        good, 4.74047 * df.pm_total_masyr / np.where(good, df.parallax, np.nan), np.nan
    )
    return df.sort_values("pm_total_masyr", ascending=False).reset_index(drop=True)


def neighborhood_figure(
    tbl: Table,
    center: SkyCoord,
    years: list[float],
    target_track: list[SkyCoord] | None = None,
    highlight_pm_masyr: float = 100.0,
) -> go.Figure:
    """Animated sky map (East left, North up): stars at each epoch, motion vectors for fast movers."""
    gmag = np.asarray(tbl["phot_g_mean_mag"], dtype=float)
    size = np.clip(14 - 0.9 * (gmag - 6), 3, 16)

    def offsets(year: float):
        c = propagate(tbl, Time(year, format="jyear"))
        dx, dy = center.spherical_offsets_to(c)
        return dx.arcsec, dy.arcsec

    x0, y0 = offsets(years[0])
    x1, y1 = offsets(years[-1])
    pm = np.hypot(
        np.nan_to_num(np.asarray(tbl["pmra"], dtype=float)),
        np.nan_to_num(np.asarray(tbl["pmdec"], dtype=float)),
    )
    fast = pm >= highlight_pm_masyr
    seg_x, seg_y = [], []
    for a, b, c_, d, f in zip(x0, x1, y0, y1, fast):
        if f:
            seg_x += [a, b, None]
            seg_y += [c_, d, None]

    def star_trace(year):
        x, y = offsets(year)
        return go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Gaia stars",
            marker=dict(
                size=size,
                color=np.log10(pm + 1),
                colorscale="Viridis",
                cmin=0,
                cmax=4,
                colorbar=dict(title="log10 PM<br>(mas/yr)"),
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"Gaia {s}<br>G={g:.1f}<br>PM={p:.0f} mas/yr"
                for s, g, p in zip(tbl["source_id"], gmag, pm)
            ],
            hoverinfo="text",
        )

    vec = go.Scatter(
        x=seg_x,
        y=seg_y,
        mode="lines",
        line=dict(width=1, color="rgba(127,227,255,0.7)"),
        name=f"Motion {years[0]:.0f}→{years[-1]:.0f} (PM ≥ {highlight_pm_masyr:.0f} mas/yr)",
        hoverinfo="skip",
    )
    data = [star_trace(years[0]), vec]
    if target_track:
        tx, ty = center.spherical_offsets_to(
            SkyCoord([c.ra for c in target_track], [c.dec for c in target_track])
        )
        data.append(
            go.Scatter(
                x=tx.arcsec,
                y=ty.arcsec,
                mode="lines+markers",
                name="Target path",
                line=dict(color="#ff6b35", width=3),
                marker=dict(symbol="x", size=7),
            )
        )
    fig = go.Figure(
        data=data, frames=[go.Frame(data=[star_trace(y)], traces=[0], name=f"{y:g}") for y in years]
    )
    fig.update_layout(
        template="plotly_dark",
        height=620,
        xaxis=dict(title="ΔRA·cosδ (arcsec, East ←)", autorange="reversed"),
        yaxis=dict(title="ΔDec (arcsec, North ↑)", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=1.08),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0,
                y=-0.12,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Epoch: "),
                steps=[
                    dict(
                        method="animate",
                        label=f"{y:g}",
                        args=[
                            [f"{y:g}"],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True)),
                        ],
                    )
                    for y in years
                ],
            )
        ],
    )
    return fig
