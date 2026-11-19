"""gta6-trophy-rarity-forecast-01 entry: forecast each of GTA VI's 55 PS5
trophies' Exophase earn rate at the case's target.

Model (written for this competition; Python 3.13, standard library only)
-------------------------------------------------------------------------
An Exophase rate is earned/tracked, and Exophase refreshes its counts on its
own schedule, so consecutive reads are often identical: the history is first
collapsed to distinct snapshots (one per refresh the house observed).

Earned counts and tracked players are modelled SEPARATELY, then combined:

    rate_i(target) = E_i(target) / T(target)

* Tracked players T: log-log power law in hours since release,
  ln T = c + d_T * ln(1 + h), fitted by recency-weighted least squares over
  the distinct snapshots and anchored at the last snapshot.  When the
  history is flat (no refresh seen yet) the slope falls back to the
  launch-night prior d_T = 1, i.e. T grows about like (1 + hours): on launch
  night the page went 638 -> 3,868 tracked in under five hours.
* Earned counts E_i: a cohort model.  Players join over time (the fitted
  smooth T curve); a player of age a has earned trophy i with probability
  A_i * (1 - exp(-a / tau_i)).  For each trophy, tau_i is picked from a grid
  by least squares on the observed earned-count increments since the first
  snapshot and A_i is the through-origin fit; A_i is shrunk toward its
  tier's median.  The forecast anchors at the last observed earned count
  and adds A_i times the cohort integral's growth to the target.  This
  captures the two forces a plain trend misses: new players dilute the
  rate, while ageing cohorts earn their way toward A_i, so early trophies
  can rise and late ones fall.
* Spread: quantiles on the logit scale around the median rate, with sigma
  growing with the log-time gap to the target, counting noise on small
  earned counts, the cohort fit's residual, and a wide band when the
  history is flat (a refresh the model has never seen can move every
  rate a lot).

Deterministic: no randomness, no wall-clock reads, fixed iteration order,
half-up rounding to 4 decimals.  Reads every *.in in $MAKEFUN_INPUT_DIR
(else /input), writes /output/<id>.out (or $MAKEFUN_OUTPUT_DIR).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (we ship none; this is just the
# same harmless line the starter carries).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuned constants (calibrated on a synthetic launch-week backtest) ---
HALF_LIFE_H = 24.0      # recency weight half-life for the fits
T_WINDOW_H = 36.0       # tracked-growth fit uses only the last 36h of snapshots
D_T_PRIOR = 1.0         # tracked-growth slope when the history is flat
D_T_MAX = 3.0           # clamp on the tracked-growth slope
# Tier priors for the cohort model, used only when a trophy has no signal
# (flat history or no earned growth observed): eventual earn rate A and
# time constant tau (hours) for a player to earn the trophy.
TIER_PRIOR = {
    "platinum": (0.01, 300.0),
    "gold": (0.35, 60.0),
    "silver": (0.22, 80.0),
    "bronze": (0.30, 60.0),
}
TAU_GRID = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0,
            120.0, 200.0, 350.0, 600.0, 1000.0)
SHRINK_K = 3.0          # pseudo-count shrinking a trophy's A to its tier median
SIGMA_K = 0.30          # spread per unit log-time gap to the target
SIGMA_BASE = 0.15       # spread floor (logit units)
SIGMA_FLAT = 1.0        # extra spread when no refresh has been seen yet
SIGMA_RARE = 0.5        # counting-noise scale: SIGMA_RARE / sqrt(E_last)
STEP_MIN = 10           # cohort-integration step, minutes


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (as JavaScript's Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    """ISO 8601 UTC with optional fractional seconds."""
    return datetime.strptime(s.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def wlsq(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, sum_w)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, sw)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    s2 = sum(w * e * e for w, e in zip(ws, resid)) / dof
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    return (a, b, se_b, sw)


def distinct_snapshots(case):
    """Collapse the history to one snapshot per observed refresh.

    Returns [(hours, minutes, players_tracked, [earned...])] oldest first,
    keeping the FIRST read that showed each distinct page.
    """
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    snaps = []
    seen = set()
    n_tr = len(case["trophies"])
    for read in case["history"]:
        rates = list(read["rates"]) + [None] * max(0, n_tr - len(read["rates"]))
        key = (read["players_tracked"], tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        when = parse_time(read["at"])
        h = (when - release).total_seconds() / 3600.0
        mins = (when - release).total_seconds() / 60.0
        tracked = read["players_tracked"] or 0
        earned = [
            (r * tracked / 100.0) if (r is not None and tracked > 0) else None
            for r in rates
        ]
        snaps.append((h, mins, tracked, earned))
    return snaps


def cohort_growth(min_first, min_last, min_target, t_first, t_last, d_t,
                  h_last, tau):
    """G(t; tau) at the snapshot minutes and at the target, for one tau.

    G(t) = sum of dT(s) * (1 - exp(-(t-s)/tau)) over cohorts joining after
    the first snapshot: the earned-count growth a trophy with eventual rate
    A=1 would show.  T follows the fitted smooth curve anchored at the last
    snapshot.  Returns {minute: G}.
    """
    want = set()
    # caller passes the minutes it needs via min_last/min_target; snapshots
    # are handled by the caller through the returned dict keyed by minute.
    step = STEP_MIN
    m = min_first
    # Smooth T at minute m: anchored log-log curve (hours since release).
    def t_smooth(minute):
        h = minute / 60.0
        return t_last * math.exp(d_t * (math.log1p(max(0.0, h))
                                        - math.log1p(max(0.0, h_last))))
    mass = 0.0          # sum dT(s) * exp(-(t-s)/tau): not-yet-earned potential
    t_prev = t_first
    out = {}
    decay = math.exp(-step / 60.0 / tau)
    # Record G at the first snapshot (0 by construction).
    out[min_first] = 0.0
    m += step
    while m <= min_target:
        t_now = t_smooth(m)
        dt_new = max(0.0, t_now - t_prev)
        mass = mass * decay + dt_new
        g = (t_now - t_first) - mass
        out[m] = max(0.0, g)
        t_prev = t_now
        m += step
    # The target minute may not fall on the grid: advance in one short step.
    if min_target not in out:
        dt_min = min_target - (m - step)
        if dt_min > 0:
            t_now = t_smooth(min_target)
            dt_new = max(0.0, t_now - t_prev)
            mass = mass * math.exp(-dt_min / 60.0 / tau) + dt_new
            out[min_target] = max(0.0, (t_now - t_first) - mass)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    n = len(trophies)
    if not case["history"]:
        return {"trophies": [
            {"id": t["id"], "q": [round4(sigmoid(logit(1.0) + z * 1.5)) for z in Z]}
            for t in trophies
        ]}

    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    h_target = (parse_time(case["target"]) - release).total_seconds() / 3600.0
    min_target = int(round((parse_time(case["target"]) - release).total_seconds() / 60.0))
    snaps = distinct_snapshots(case)
    h_last, min_last, t_last, _e = snaps[-1]
    h_first, min_first, t_first, _e0 = snaps[0]
    x_target = math.log1p(max(0.0, h_target))
    x_last = math.log1p(max(0.0, h_last))
    dx = max(0.0, x_target - x_last)
    flat = len(snaps) < 2

    # --- tracked players: log-log fit, anchored at the last snapshot ---
    # Only the recent window: launch-week growth accelerates, so the early
    # (slow) phase would drag the slope down and understate the target.
    xs, ys, ws = [], [], []
    for h, _m, t, _e in snaps:
        if t <= 0 or h < h_last - T_WINDOW_H:
            continue
        xs.append(math.log1p(max(0.0, h)))
        ys.append(math.log(t))
        ws.append(0.5 ** ((h_last - h) / HALF_LIFE_H))
    if len(xs) < 2:
        xs, ys, ws = [], [], []
        for h, _m, t, _e in snaps:
            if t <= 0:
                continue
            xs.append(math.log1p(max(0.0, h)))
            ys.append(math.log(t))
            ws.append(0.5 ** ((h_last - h) / HALF_LIFE_H))
    fit_t = wlsq(xs, ys, ws)
    d_t = D_T_PRIOR if (fit_t is None or flat) else min(D_T_MAX, max(0.0, fit_t[1]))
    t_target = t_last * math.exp(d_t * dx)
    t_target = min(t_target, t_last * 50.0 + 1.0)

    # --- cohort integrals G(t; tau) on the snapshot grid ---
    snap_minutes = [int(round(s[1])) for s in snaps]
    g_tables = {}
    for tau in TAU_GRID:
        g = cohort_growth(min_first, min_last, min_target, t_first, t_last,
                          d_t, h_last, tau)
        # Snapshots may fall between grid steps; G is monotone, so take the
        # nearest recorded minute at or before the snapshot.
        g_tables[tau] = g

    def g_at(table, minute):
        if minute in table:
            return table[minute]
        below = [m for m in table if m <= minute]
        return table[max(below)] if below else 0.0

    # --- per-trophy cohort fit: pick tau, fit A, shrink A to the tier median ---
    weights = [0.5 ** ((h_last - s[0]) / HALF_LIFE_H) for s in snaps]
    fitted = []
    for i in range(n):
        e_first = snaps[0][3][i]
        if e_first is None:
            e_first = 0.0
        d_e, g_rows, w_rows = [], [], []
        for j, s in enumerate(snaps):
            ei = s[3][i]
            if ei is None:
                continue
            d_e.append(ei - e_first)
            g_rows.append(snap_minutes[j])
            w_rows.append(weights[j])
        if flat or len(d_e) < 3 or max(d_e) <= 0:
            fitted.append(None)
            continue
        cands = []
        for tau in TAU_GRID:
            table = g_tables[tau]
            gs = [g_at(table, mm) for mm in g_rows]
            num = sum(w * de * gg for w, de, gg in zip(w_rows, d_e, gs))
            den = sum(w * gg * gg for w, gg in zip(w_rows, gs))
            if den <= 0:
                continue
            a_fit = num / den
            ss = sum(w * (de - a_fit * gg) ** 2 for w, de, gg in zip(w_rows, d_e, gs))
            cands.append((ss, tau, a_fit))
        # Among near-best fits, take the LARGEST tau: (A, tau) trade off, and
        # when the data cannot tell them apart the slow-saturating reading
        # is the conservative one (it does not invent fast earned growth).
        min_ss = min(c[0] for c in cands)
        near = [c for c in cands if c[0] <= 1.10 * min_ss]
        ss, tau, a_fit = max(near, key=lambda c: c[1])
        fitted.append((ss, tau, a_fit, sum(w_rows)))
    # Tier median A among trophies with a real fit, for shrinkage.
    tier_a = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = sorted(f[2] for f, tr in zip(fitted, trophies)
                      if f is not None and tr["tier"] == tier and f[2] > 0)
        if vals:
            tier_a[tier] = vals[len(vals) // 2]
    for tier in ("platinum", "gold", "silver", "bronze"):
        tier_a.setdefault(tier, TIER_PRIOR[tier][0])

    rows = []
    for i, tr in enumerate(trophies):
        tier = tr["tier"]
        fit = fitted[i]
        # Last known earned count for this trophy (walk back over snapshots).
        e_last_i = None
        for s in reversed(snaps):
            if s[3][i] is not None:
                e_last_i = s[3][i]
                break
        if e_last_i is None or e_last_i <= 0:
            e_last_i = 0.5
        if fit is None:
            a_prior, tau_prior = TIER_PRIOR.get(tier, (0.30, 60.0))
            # A trophy already at ~100% is earned on joining: instant, not
            # after the tier's time constant (launch night: the 100% trophy
            # stayed 100% through the first refresh while every other rate fell).
            if e_last_i >= 0.99 * max(1, t_last):
                a_prior, tau_prior = 1.0, 0.1
            a_fit = a_prior
            # Snap the prior tau onto the grid the G tables were built on.
            tau_fit = min(TAU_GRID, key=lambda t: abs(math.log(t / tau_prior)))
            ss = None
            n_eff = 0.0
        else:
            ss, tau_fit, a_fit, n_eff = fit
            a_fit = (n_eff * a_fit + SHRINK_K * tier_a[tier]) / (n_eff + SHRINK_K)
        a_fit = min(1.0, max(0.0, a_fit))
        # A trophy's rate can never exceed its eventual rate A, so A is at
        # least the highest rate ever observed for it.
        rate_max = 0.0
        for _h, _m, t, e in snaps:
            if e[i] is not None and t > 0:
                rate_max = max(rate_max, e[i] / t)
        a_fit = max(a_fit, rate_max)
        table = g_tables[tau_fit]
        g_last = g_at(table, min_last)
        g_tgt = g_at(table, min_target)
        e_target = e_last_i + a_fit * max(0.0, g_tgt - g_last)
        # Earned can never exceed the eventual rate times tracked players.
        e_target = min(e_target, a_fit * t_target) if a_fit > 0 else e_target
        e_target = max(e_target, e_last_i)
        rate = 100.0 * e_target / t_target if t_target > 0 else 0.01
        rate = clamp(rate)
        # --- spread ---
        var = (SIGMA_K * dx) ** 2 + SIGMA_BASE ** 2
        var += (SIGMA_RARE / math.sqrt(max(1.0, e_last_i))) ** 2
        if flat:
            var += SIGMA_FLAT ** 2
        if ss is not None and n_eff > 0:
            resid = math.sqrt(ss / max(1.0, n_eff))
            var += (0.5 * resid / max(1.0, e_last_i)) ** 2
        sigma = math.sqrt(var)
        lg = logit(rate)
        qs = [round4(sigmoid(lg + z * sigma)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
