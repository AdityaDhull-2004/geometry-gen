"""Numeric claim-verification helpers for the logic audit.

Probe a construction's POINT coordinates in headless GeoGebra across several
random renders, then test whether the property the PROBLEM STATEMENT asserts
actually holds. Rendering proves a construction *builds*; these helpers prove it
*means what the statement says*.

Typical use inside an audit:

    from audit_helpers import coords_trials, collinear, concyclic, eqlen, perp, dist
    trials = coords_trials(steps, n=6)          # list of {name:[x,y]} dicts
    ok = all(collinear(c['X'], c['Y'], c['Z']) for c in trials)   # Simson line

All predicates return bool and use generous tolerances so a TRUE theorem passes
on every random trial; a FALSE claim will fail on at least one trial.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator import GeoGebraValidator


def coords_trials(steps, n=6):
    """Return a list of n coord dicts (name -> [x, y]) from n fresh random renders.
    Skips trials where the build failed."""
    out = []
    with GeoGebraValidator() as v:
        for _ in range(n):
            failed, coords = v.probe(steps, random_t=True)
            if not failed:
                out.append(coords)
    return out


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eq(x, y, tol=1e-4):
    return abs(x - y) <= tol * max(1.0, abs(x), abs(y))


def eqlen(a, b, c, d, tol=1e-4):
    """|AB| == |CD|."""
    return eq(dist(a, b), dist(c, d), tol)


def collinear(a, b, c, tol=1e-6):
    """A, B, C collinear (twice signed area ~ 0, scaled by spread)."""
    area2 = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    scale = max(dist(a, b), dist(b, c), dist(a, c), 1.0)
    return area2 / (scale*scale) < tol or area2 < 1e-4


def _circumcenter(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by))/d
    uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax))/d
    return (ux, uy)


def concyclic(a, b, c, d, tol=1e-3):
    """A, B, C, D lie on one circle: D equidistant from circumcenter of ABC."""
    o = _circumcenter(a, b, c)
    if o is None:
        return collinear(a, b, c) and collinear(a, b, d)
    r = dist(o, a)
    return abs(dist(o, d) - r) <= tol * max(1.0, r)


def on_circle(p, o, r, tol=1e-3):
    return abs(dist(p, o) - r) <= tol * max(1.0, r)


def perp(a, b, c, d, tol=1e-4):
    """Line AB perpendicular to line CD (dot of direction vectors ~ 0)."""
    u = (b[0]-a[0], b[1]-a[1]); w = (d[0]-c[0], d[1]-c[1])
    nu = math.hypot(*u); nw = math.hypot(*w)
    if nu < 1e-9 or nw < 1e-9:
        return False
    return abs(u[0]*w[0] + u[1]*w[1])/(nu*nw) < tol


def parallel(a, b, c, d, tol=1e-4):
    u = (b[0]-a[0], b[1]-a[1]); w = (d[0]-c[0], d[1]-c[1])
    nu = math.hypot(*u); nw = math.hypot(*w)
    if nu < 1e-9 or nw < 1e-9:
        return False
    return abs(u[0]*w[1] - u[1]*w[0])/(nu*nw) < tol


def ratio(a, b, c, target, tol=1e-3):
    """|AB|/|BC| == target."""
    bc = dist(b, c)
    if bc < 1e-9:
        return False
    return eq(dist(a, b)/bc, target, tol)


def concurrent(p1a, p1b, p2a, p2b, p3a, p3b, tol=1e-3):
    """Lines (p1a,p1b), (p2a,p2b), (p3a,p3b) share a common point."""
    def inter(a, b, c, d):
        x1,y1=a; x2,y2=b; x3,y3=c; x4,y4=d
        den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(den)<1e-12: return None
        px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
        py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
        return (px,py)
    i12 = inter(p1a,p1b,p2a,p2b)
    i13 = inter(p1a,p1b,p3a,p3b)
    if i12 is None or i13 is None:
        return False
    return dist(i12, i13) < tol * max(1.0, dist(p1a,p1b))
