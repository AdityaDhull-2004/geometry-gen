"""Numeric geometry checks evaluated on probed point coordinates.

A candidate problem may carry a top-level `checks` list asserting the theorem it
illustrates, e.g.

    "checks": [
        {"type": "parallel",   "a": ["P", "Q"], "b": ["B", "C"]},
        {"type": "concyclic",  "points": ["A", "B", "C", "D"]},
        {"type": "concurrent", "lines": [["A","X"], ["B","Y"], ["C","Z"]]},
        {"type": "distinct",   "points": ["M", "R"]}
    ]

Each check is verified against the FULL-PRECISION coordinates probed from GeoGebra,
on every random-t trial. Because true theorems hold for all positions while a wrong
intersection-index pick is off by O(1), a tolerance of 1e-6 cleanly separates them.

Supported types:
  collinear  {points:[...]}                three+ points on one line
  concyclic  {points:[...]}                four+ points on one circle
  concurrent {lines:[[p,q],...]}           three+ lines through one point
  parallel   {a:[p,q], b:[r,s]}            two segments/lines parallel
  perpendicular {a:[p,q], b:[r,s]}         two segments/lines perpendicular
  equal_len  {a:[p,q], b:[r,s]}            |pq| == |rs|
  on_line    {point:p, line:[a,b]}         p on line ab
  on_circle  {point:p, circle:[a,b,c]}     p on circle through a,b,c
  distinct   {points:[...]}                all pairwise distinct (anti-degeneracy)
  midpoint   {point:m, of:[a,b]}           m == midpoint(a,b)
"""
import math

TOL = 1e-6          # relative tolerance for "equal" / "on"
DISTINCT_MIN = 1e-3  # points closer than this count as coincident


def _sub(p, q): return (p[0] - q[0], p[1] - q[1])
def _cross(u, v): return u[0] * v[1] - u[1] * v[0]
def _dot(u, v): return u[0] * v[0] + u[1] * v[1]
def _norm(u): return math.hypot(u[0], u[1])
def _dist(p, q): return math.hypot(p[0] - q[0], p[1] - q[1])


def _angle(v, p, q):
    """Unsigned angle (degrees) at vertex v between rays v->p and v->q."""
    u, w = _sub(p, v), _sub(q, v)
    nu, nw = _norm(u), _norm(w)
    if nu < 1e-12 or nw < 1e-12:
        return None
    c = max(-1.0, min(1.0, _dot(u, w) / (nu * nw)))
    return math.degrees(math.acos(c))


def _scale(coords):
    """Characteristic length of the figure, for relative tolerances."""
    pts = list(coords.values())
    if len(pts) < 2:
        return 1.0
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    s = max(max(xs) - min(xs), max(ys) - min(ys))
    return s if s > 1e-9 else 1.0


def _circle_from3(a, b, c):
    """Centre & radius of circle through a,b,c; None if (near) collinear."""
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    centre = (ux, uy)
    return centre, _dist(centre, a)


def _line_intersect(a, b, c, d):
    """Intersection of line ab and line cd; None if parallel."""
    r = _sub(b, a); s = _sub(d, c)
    den = _cross(r, s)
    if abs(den) < 1e-12:
        return None
    t = _cross(_sub(c, a), s) / den
    return (a[0] + t * r[0], a[1] + t * r[1])


def evaluate(assertions, coords):
    """Return (ok, reason). reason is None on success, else a short string."""
    scale = _scale(coords)
    tol = TOL * scale

    def P(name):
        if name not in coords:
            raise KeyError(name)
        return coords[name]

    for chk in assertions:
        t = chk.get("type")
        try:
            if t == "collinear":
                pts = [P(n) for n in chk["points"]]
                a, b = pts[0], pts[1]
                base = _norm(_sub(b, a))
                if base < tol:
                    return False, f"collinear: {chk['points'][0]}={chk['points'][1]}"
                for n, p in zip(chk["points"][2:], pts[2:]):
                    if abs(_cross(_sub(b, a), _sub(p, a))) / base > tol:
                        return False, f"collinear: {n} off line"

            elif t == "concyclic":
                pts = [P(n) for n in chk["points"]]
                circ = _circle_from3(pts[0], pts[1], pts[2])
                if circ is None:
                    return False, "concyclic: first 3 collinear"
                centre, rad = circ
                for n, p in zip(chk["points"][3:], pts[3:]):
                    if abs(_dist(centre, p) - rad) > tol:
                        return False, f"concyclic: {n} off circle"

            elif t == "concurrent":
                ls = chk["lines"]
                a, b = P(ls[0][0]), P(ls[0][1])
                c, d = P(ls[1][0]), P(ls[1][1])
                X = _line_intersect(a, b, c, d)
                if X is None:
                    return False, "concurrent: first two parallel"
                for ln in ls[2:]:
                    u, v = P(ln[0]), P(ln[1])
                    base = _norm(_sub(v, u))
                    if base < tol or abs(_cross(_sub(v, u), _sub(X, u))) / base > tol:
                        return False, f"concurrent: line {ln} misses point"

            elif t == "parallel":
                u = _sub(P(chk["a"][1]), P(chk["a"][0]))
                v = _sub(P(chk["b"][1]), P(chk["b"][0]))
                if _norm(u) < tol or _norm(v) < tol:
                    return False, "parallel: degenerate segment"
                if abs(_cross(u, v)) / (_norm(u) * _norm(v)) > TOL:
                    return False, "parallel: not parallel"

            elif t == "perpendicular":
                u = _sub(P(chk["a"][1]), P(chk["a"][0]))
                v = _sub(P(chk["b"][1]), P(chk["b"][0]))
                if _norm(u) < tol or _norm(v) < tol:
                    return False, "perpendicular: degenerate segment"
                if abs(_dot(u, v)) / (_norm(u) * _norm(v)) > TOL:
                    return False, "perpendicular: not perpendicular"

            elif t == "equal_len":
                la = _dist(P(chk["a"][0]), P(chk["a"][1]))
                lb = _dist(P(chk["b"][0]), P(chk["b"][1]))
                if abs(la - lb) > tol:
                    return False, f"equal_len: {la:.4f} != {lb:.4f}"

            elif t == "on_line":
                a, b = P(chk["line"][0]), P(chk["line"][1])
                p = P(chk["point"])
                base = _norm(_sub(b, a))
                if base < tol or abs(_cross(_sub(b, a), _sub(p, a))) / base > tol:
                    return False, f"on_line: {chk['point']} off line"

            elif t == "on_circle":
                circ = _circle_from3(P(chk["circle"][0]), P(chk["circle"][1]), P(chk["circle"][2]))
                if circ is None:
                    return False, "on_circle: defining points collinear"
                centre, rad = circ
                if abs(_dist(centre, P(chk["point"])) - rad) > tol:
                    return False, f"on_circle: {chk['point']} off circle"

            elif t == "distinct":
                names = chk["points"]
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        if _dist(P(names[i]), P(names[j])) < DISTINCT_MIN * scale:
                            return False, f"distinct: {names[i]}={names[j]}"

            elif t == "tangent_circles":
                # c1/c2 given as [centre_name, point_on_circle_name]; internal flag
                c1c, c1p = P(chk["c1"][0]), P(chk["c1"][1])
                c2c, c2p = P(chk["c2"][0]), P(chk["c2"][1])
                r1 = _dist(c1c, c1p); r2 = _dist(c2c, c2p); d = _dist(c1c, c2c)
                target = abs(r1 - r2) if chk.get("internal") else (r1 + r2)
                if abs(d - target) > tol:
                    return False, f"tangent_circles: d={d:.4f} target={target:.4f}"

            elif t == "equal_area":
                def _area(a, b, c):
                    return abs((b[0]-a[0]) * (c[1]-a[1]) - (c[0]-a[0]) * (b[1]-a[1])) / 2
                areas = [_area(P(tr[0]), P(tr[1]), P(tr[2])) for tr in chk["tris"]]
                a0 = areas[0]
                for ar in areas[1:]:
                    if abs(ar - a0) > TOL * max(a0, ar, 1e-12):
                        return False, f"equal_area: {a0:.5f} != {ar:.5f}"

            elif t == "coincide":
                a, b = P(chk["points"][0]), P(chk["points"][1])
                if _dist(a, b) > tol:
                    return False, f"coincide: dist {_dist(a, b):.4f}"

            elif t == "midpoint":
                m = P(chk["point"]); a, b = P(chk["of"][0]), P(chk["of"][1])
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if _dist(m, mid) > tol:
                    return False, f"midpoint: {chk['point']} not midpoint"

            elif t == "len_product_equal":
                prod1 = 1.0
                for pr in chk["pairs1"]:
                    prod1 *= _dist(P(pr[0]), P(pr[1]))
                prod2 = 1.0
                for pr in chk["pairs2"]:
                    prod2 *= _dist(P(pr[0]), P(pr[1]))
                if abs(prod1 - prod2) > TOL * max(prod1, prod2, 1e-12):
                    return False, f"len_product_equal: {prod1:.5f} != {prod2:.5f}"

            elif t == "angle_value":
                ang = _angle(P(chk["points"][1]), P(chk["points"][0]), P(chk["points"][2]))
                if ang is None:
                    return False, "angle_value: degenerate"
                if abs(ang - chk["deg"]) > 1e-2:
                    return False, f"angle_value: {ang:.4f} != {chk['deg']}"

            elif t == "angle_equal":
                a1 = _angle(P(chk["a"][1]), P(chk["a"][0]), P(chk["a"][2]))
                a2 = _angle(P(chk["b"][1]), P(chk["b"][0]), P(chk["b"][2]))
                if a1 is None or a2 is None:
                    return False, "angle_equal: degenerate"
                if abs(a1 - a2) > 1e-2:
                    return False, f"angle_equal: {a1:.4f} != {a2:.4f}"

            elif t == "angle_double":
                big = _angle(P(chk["big"][1]), P(chk["big"][0]), P(chk["big"][2]))
                small = _angle(P(chk["small"][1]), P(chk["small"][0]), P(chk["small"][2]))
                if big is None or small is None:
                    return False, "angle_double: degenerate"
                if abs(big - 2 * small) > 1e-2:
                    return False, f"angle_double: {big:.3f} != 2*{small:.3f}"

            elif t == "len_sum_equal":
                s1 = sum(_dist(P(pr[0]), P(pr[1])) for pr in chk["a"])
                s2 = sum(_dist(P(pr[0]), P(pr[1])) for pr in chk["b"])
                if abs(s1 - s2) > TOL * max(s1, s2, 1e-12):
                    return False, f"len_sum_equal: {s1:.5f} != {s2:.5f}"

            elif t == "sum_sq_equal":
                s1 = sum(_dist(P(pr[0]), P(pr[1])) ** 2 for pr in chk["a"])
                s2 = sum(_dist(P(pr[0]), P(pr[1])) ** 2 for pr in chk["b"])
                if abs(s1 - s2) > TOL * max(s1, s2, 1e-12):
                    return False, f"sum_sq_equal: {s1:.5f} != {s2:.5f}"

            elif t == "ptolemy":
                a, b, c, d = (P(n) for n in chk["points"])  # cyclic order A,B,C,D
                lhs = _dist(a, c) * _dist(b, d)
                rhs = _dist(a, b) * _dist(c, d) + _dist(a, d) * _dist(b, c)
                if abs(lhs - rhs) > TOL * max(lhs, rhs, 1e-12):
                    return False, f"ptolemy: {lhs:.5f} != {rhs:.5f}"

            else:
                return False, f"unknown check type: {t}"

        except KeyError as e:
            return False, f"{t}: missing point {e}"

    return True, None
