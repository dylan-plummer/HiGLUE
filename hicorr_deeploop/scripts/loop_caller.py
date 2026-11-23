r"""(Copy of islet_benchmark/loop_caller.py, so the pipeline is self-contained; keep the two in sync.)
Loop calling on DeepLoop2 output (anchor-level coolers).

Default (``call_loops_clusters``, chosen 2026-11-17): the ``n_pixels`` (500,000) strongest band pixels genome-wide
(``MIN_D``-``MAX_D``), clustered by adjacency (8-connected on the anchor grid); one loop per cluster at its
value-weighted centre of mass, snapped to the nearest anchors. ``value`` = the cluster's strongest pixel, ``npix`` =
its size. Unlike the callers below it neither tiles stripes nor needs a prominence rule: a blob is one loop wherever
its pixels pass the genome-wide cutoff.

Peak caller (``call_loops_peaks``, used 2026-11-16 only):
DeepLoop2 returns each loop as a neighbourhood of enhanced pixels, typically ~30 x 30 kb, so taking the top pixels
with a small non-maximum suppression spends most of the budget on the shoulders of strong loops (on the Beta pool,
87% of top-50k pixel calls shared a blob with another call). Here a loop is one summit per peak:

1. Region: band pixels at ``MIN_D``-``MAX_D`` with value >= the ``floor_q`` quantile of the chromosome's band
   (chrY / chrM not called, see ``EXCLUDE``).
2. Summits and prominence (0-dimensional persistence on the 8-connected pixel grid): pixels are added from the
   highest down; a pixel touching no earlier pixel starts a peak; a pixel joining several peaks is a saddle, where
   every peak but the highest ends with ``prom = log2(summit / saddle)``. Peaks still alive at the floor get
   ``log2(summit / floor)``. A peak is a loop when ``prom >= min_prom``, so two summits in one blob are both kept
   only when the valley between them is deep enough.
3. Merge: summits within ``merge_bp`` of a stronger summit on both ends (anchor midpoints) are dropped, which makes
   the separation genomic rather than counted in anchors of variable size.
4. The ``n`` strongest summits genome-wide, at least ``min_off`` anchors off the diagonal.

``summits()`` does steps 1-2 once per cooler (the slow part); ``select()`` does 3-4 for any parameters.
"""
import numpy as np
import pandas as pd

MIN_D, MAX_D = 20_000, 2_000_000
# not called: chrY's band is near-empty in mixed-sex pools, so its per-chromosome floor is tiny and noise passes as
# peaks (found on the islet pools); chrM / contigs have no band worth calling
EXCLUDE = ("chrY", "chrM", "chrMT", "Y", "M", "MT")
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _persistence(b1, b2, v, floor):
    r"""summit index -> (prominence in log2, saddle value) for the pixels (b1, b2, v) of one chromosome"""
    n = len(v)
    key = b1.astype(np.int64) * (1 << 24) + b2.astype(np.int64)
    order_key = np.argsort(key)
    sk = key[order_key]
    nbr = np.full((n, 8), -1, dtype=np.int64)
    for k, (di, dj) in enumerate(OFFSETS):
        q = key + di * (1 << 24) + dj
        pos = np.searchsorted(sk, q)
        pos[pos >= n] = 0
        hit = sk[pos] == q
        nbr[hit, k] = order_key[pos[hit]]
    order = np.argsort(-v, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    parent = np.full(n, -1, dtype=np.int64)  # union-find over processed pixels
    peak = np.full(n, -1, dtype=np.int64)    # root -> its summit pixel
    prom, saddle = {}, {}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    nbr_rank = np.where(nbr >= 0, rank[np.maximum(nbr, 0)], n)
    for p in order:
        rp = rank[p]
        roots = {find(q) for q, rq in zip(nbr[p], nbr_rank[p]) if rq < rp}
        if not roots:
            parent[p] = p
            peak[p] = p
            continue
        win = min(roots, key=lambda r: rank[peak[r]])  # highest summit survives (elder rule)
        for r in roots:
            if r != win:
                s = peak[r]
                prom[s] = np.log2(v[s] / v[p])
                saddle[s] = v[p]
                parent[r] = win
        parent[p] = win
    for r in np.unique([find(x) for x in np.flatnonzero(parent >= 0)]):
        s = peak[r]
        prom[s] = np.log2(v[s] / floor)
        saddle[s] = floor
    idx = np.fromiter(prom.keys(), dtype=np.int64)
    return idx, np.fromiter(prom.values(), dtype=float), np.array([saddle[i] for i in idx])


def summits(clr, floor_q: float = 0.98, chroms=None) -> pd.DataFrame:
    r"""every peak of the DeepLoop2 cooler with its prominence: chrom, bin1, bin2, pos1, pos2, value, prom, saddle,
    floor"""
    bins = clr.bins()[["chrom", "start", "end"]][:]
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    out = []
    for c in chroms or [c for c in clr.chromnames if c not in EXCLUDE]:
        px = clr.pixels().fetch(c)
        if px.empty:
            continue
        b1, b2, v = px["bin1_id"].to_numpy(), px["bin2_id"].to_numpy(), px["count"].to_numpy().astype(float)
        d = mid[b2] - mid[b1]
        band = (d >= MIN_D) & (d <= MAX_D) & (b2 > b1)
        if band.sum() < 100:
            continue
        floor = float(np.quantile(v[band], floor_q))
        keep = band & (v >= floor)
        b1, b2, v = b1[keep], b2[keep], v[keep]
        idx, pr, sd = _persistence(b1, b2, v, floor)
        out.append(pd.DataFrame({"chrom": c, "bin1": b1[idx], "bin2": b2[idx], "pos1": mid[b1[idx]],
                                 "pos2": mid[b2[idx]], "value": v[idx], "prom": pr, "saddle": sd, "floor": floor}))
    return pd.concat(out, ignore_index=True)


def merge(s: pd.DataFrame, merge_bp: int) -> pd.DataFrame:
    r"""drop summits within ``merge_bp`` (both ends) of a stronger kept summit"""
    if merge_bp <= 0 or s.empty:
        return s
    s = s.sort_values("value", ascending=False)
    keep = np.zeros(len(s), bool)
    grid = {}
    for i, (c, p1, p2) in enumerate(zip(s["chrom"].to_numpy(), s["pos1"].to_numpy(), s["pos2"].to_numpy())):
        g1, g2 = p1 // merge_bp, p2 // merge_bp
        clash = False
        for a in (g1 - 1, g1, g1 + 1):
            for b in (g2 - 1, g2, g2 + 1):
                for q1, q2 in grid.get((c, a, b), ()):
                    if abs(q1 - p1) <= merge_bp and abs(q2 - p2) <= merge_bp:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep[i] = True
            grid.setdefault((c, g1, g2), []).append((p1, p2))
    return s[keep]


def select(s: pd.DataFrame, n: int, min_prom: float = 0.3, merge_bp: int = 20_000, min_off: int = 0) -> pd.DataFrame:
    r"""loops from ``summits()``: prominent summits, merged, top ``n`` by value"""
    s = s[(s["prom"] >= min_prom) & (s["bin2"] - s["bin1"] >= min_off)]
    # only the strongest candidates can make the top n; merging never removes a stronger summit
    s = merge(s.nlargest(min(len(s), 4 * n), "value"), merge_bp)
    return s.nlargest(n, "value").reset_index(drop=True)


def top_pixels(clr, per_chrom: int = 80_000, chroms=None) -> pd.DataFrame:
    r"""the strongest band pixels of each chromosome: chrom, bin1, bin2, pos1, pos2, value"""
    bins = clr.bins()[["chrom", "start", "end"]][:]
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    out = []
    for c in chroms or [c for c in clr.chromnames if c not in EXCLUDE]:
        px = clr.pixels().fetch(c)
        if px.empty:
            continue
        d = mid[px["bin2_id"].to_numpy()] - mid[px["bin1_id"].to_numpy()]
        px = px[(d >= MIN_D) & (d <= MAX_D)].nlargest(per_chrom, "count")
        b1, b2 = px["bin1_id"].to_numpy(), px["bin2_id"].to_numpy()
        out.append(pd.DataFrame({"chrom": c, "bin1": b1, "bin2": b2, "pos1": mid[b1], "pos2": mid[b2],
                                 "value": px["count"].to_numpy()}))
    return pd.concat(out, ignore_index=True)


def cluster_pixels(top: pd.DataFrame, n_pixels: int, gap: int = 1, bins: pd.DataFrame = None) -> pd.DataFrame:
    r"""the ``n_pixels`` strongest pixels genome-wide (``top``: chrom bin1 bin2 pos1 pos2 value), clustered by
    adjacency (pixels within ``gap`` anchors on both axes are one cluster, i.e. 8-connected for gap 1); one loop per
    cluster at its value-weighted centre of mass (genomic), snapped to the nearest anchor (bin1 / bin2) when the
    cooler's ``bins`` (chrom start end) are given. Returns chrom bin1 bin2 pos1 pos2 value (cluster max) sum npix span1 span2 (bp), strongest
    first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    t = top.nlargest(n_pixels, "value").reset_index(drop=True)
    out = []
    for c, g in t.groupby("chrom", sort=False):
        b1, b2 = g["bin1"].to_numpy().astype(np.int64), g["bin2"].to_numpy().astype(np.int64)
        key = b1 * (1 << 24) + b2
        o = np.argsort(key)
        sk = key[o]
        src, dst = [], []
        for di in range(0, gap + 1):
            for dj in range(-gap, gap + 1):
                if di == 0 and dj <= 0:
                    continue
                q = (b1 + di) * (1 << 24) + (b2 + dj)
                i = np.searchsorted(sk, q)
                i[i >= len(sk)] = 0
                hit = sk[i] == q
                src.append(np.flatnonzero(hit))
                dst.append(o[i[hit]])
        src, dst = np.concatenate(src), np.concatenate(dst)
        m = len(g)
        _, lab = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(m, m)), directed=False)
        v = g["value"].to_numpy()
        df = pd.DataFrame({"lab": lab, "v": v, "p1": g["pos1"].to_numpy() * v, "p2": g["pos2"].to_numpy() * v,
                           "x1": g["pos1"].to_numpy(), "x2": g["pos2"].to_numpy()})
        a = df.groupby("lab").agg(s=("v", "sum"), mx=("v", "max"), n=("v", "size"), p1=("p1", "sum"),
                                  p2=("p2", "sum"), lo1=("x1", "min"), hi1=("x1", "max"), lo2=("x2", "min"),
                                  hi2=("x2", "max"))
        out.append(pd.DataFrame({"chrom": c, "pos1": (a["p1"] / a["s"]).round().astype(np.int64),
                                 "pos2": (a["p2"] / a["s"]).round().astype(np.int64), "value": a["mx"].to_numpy(),
                                 "sum": a["s"].to_numpy(), "npix": a["n"].to_numpy(),
                                 "span1": (a["hi1"] - a["lo1"]).to_numpy(), "span2": (a["hi2"] - a["lo2"]).to_numpy()}))
    L = pd.concat(out, ignore_index=True)
    if bins is not None:
        L = snap(L, bins)
    return L.sort_values("value", ascending=False).reset_index(drop=True)


def _nearest(mid: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""index of the nearest of the sorted midpoints ``mid`` (one chromosome)"""
    i = np.clip(np.searchsorted(mid, x), 1, len(mid) - 1)
    return np.where(np.abs(mid[i - 1] - x) <= np.abs(mid[i] - x), i - 1, i)


def snap(L: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    r"""bin1 / bin2 = the anchors (of ``bins``: chrom start end, cooler order) nearest to pos1 / pos2"""
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    L = L.copy()
    L["bin1"], L["bin2"] = 0, 0
    for c, idx in L.groupby("chrom").indices.items():
        lo = int(np.flatnonzero(bins["chrom"].to_numpy() == c)[0])
        hi = lo + int((bins["chrom"] == c).sum())
        m = mid[lo:hi]
        L.iloc[idx, L.columns.get_loc("bin1")] = lo + _nearest(m, L["pos1"].to_numpy()[idx])
        L.iloc[idx, L.columns.get_loc("bin2")] = lo + _nearest(m, L["pos2"].to_numpy()[idx])
    return L


def call_loops_clusters(clr, n_pixels: int = 500_000, min_off: int = 0, gap: int = 1) -> pd.DataFrame:
    r"""loops of a DeepLoop2 cooler: the n_pixels strongest band pixels genome-wide (at least ``min_off`` anchors off
    the diagonal) -> clusters -> one loop per cluster at its centre of mass. Returns bin1 bin2 value chrom pos1 pos2
    npix span1 span2 kind stripe (strongest first); ``cutoff`` (attrs) = the n_pixels-th pixel value."""
    bins = clr.bins()[["chrom", "start", "end"]][:]
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    top = []
    for c in [c for c in clr.chromnames if c not in EXCLUDE]:
        px = clr.pixels().fetch(c)
        if px.empty:
            continue
        b1, b2 = px["bin1_id"].to_numpy(), px["bin2_id"].to_numpy()
        d = mid[b2] - mid[b1]
        px = px[(d >= MIN_D) & (d <= MAX_D) & (b2 - b1 >= min_off)].nlargest(n_pixels, "count")
        b1, b2 = px["bin1_id"].to_numpy(), px["bin2_id"].to_numpy()
        top.append(pd.DataFrame({"chrom": c, "bin1": b1, "bin2": b2, "pos1": mid[b1], "pos2": mid[b2],
                                 "value": px["count"].to_numpy()}))
        top = [pd.concat(top, ignore_index=True).nlargest(n_pixels, "value")]  # keep memory bounded
    top = top[0]
    out = cluster_pixels(top, n_pixels, gap, bins)
    out["kind"] = "cluster"
    out["stripe"] = anchor_load(out)
    out = out[["bin1", "bin2", "value", "chrom", "pos1", "pos2", "npix", "span1", "span2", "kind", "stripe"]]
    out.attrs["cutoff"] = float(top["value"].min())
    return out


def anchor_load(loops: pd.DataFrame, w: int = 1) -> np.ndarray:
    r"""per loop, the larger of its two anchors' call counts (calls with an anchor within +-w anchors)"""
    b = np.concatenate([loops["bin1"].to_numpy(), loops["bin2"].to_numpy()])
    u, c = np.unique(b, return_counts=True)
    cnt = dict(zip(u, c))
    load = np.array([sum(cnt.get(x + d, 0) for d in range(-w, w + 1)) for x in b])
    return np.maximum(load[:len(loops)], load[len(loops):])


def fill(core: pd.DataFrame, top: pd.DataFrame, n: int, fill_bp: int = 15_000, min_off: int = 0,
         cap: int = 0) -> pd.DataFrame:
    r"""core loops + the strongest other pixels, no two calls within ``fill_bp`` on both ends (and none within
    ``fill_bp`` of a core loop), up to n calls. ``kind`` = peak / fill. Fills tile what one summit per peak leaves
    out: stripes and broad DeepLoop2 blobs, where islet pcHi-C finds one promoter contacting several sites.
    ``cap`` > 0: a fill is taken only while both its anchors (+-1 anchor) carry fewer than ``cap`` calls (peaks
    included), so the budget goes to more blobs instead of running down one stripe."""
    core = core.assign(kind="peak")
    if len(core) >= n or fill_bp <= 0:
        return core.head(n)
    top = top[top["bin2"] - top["bin1"] >= min_off]
    cand = pd.concat([core.assign(value=np.inf), top.assign(kind="fill")], ignore_index=True)
    m = merge(cand, fill_bp)
    fl = m[m["kind"] == "fill"].sort_values("value", ascending=False)
    if cap <= 0:
        extra = fl.head(n - len(core))
    else:
        load = {}
        for b in np.concatenate([core["bin1"].to_numpy(), core["bin2"].to_numpy()]):
            load[b] = load.get(b, 0) + 1

        def busy(x):
            return load.get(x - 1, 0) + load.get(x, 0) + load.get(x + 1, 0) >= cap

        keep = []
        for i, b1, b2 in zip(fl.index, fl["bin1"].to_numpy(), fl["bin2"].to_numpy()):
            if busy(b1) or busy(b2):
                continue
            keep.append(i)
            load[b1] = load.get(b1, 0) + 1
            load[b2] = load.get(b2, 0) + 1
            if len(keep) == n - len(core):
                break
        extra = fl.loc[keep]
    return pd.concat([core, extra], ignore_index=True)


def call_loops_peaks(clr, n: int, min_off: int = 0, min_prom: float = 0.3, merge_bp: int = 20_000,
                     floor_q: float = 0.98, fill_bp: int = 0, fill_cap: int = 0) -> pd.DataFrame:
    r"""loops of a DeepLoop2 cooler: bin1, bin2, value, chrom, pos1, pos2 (as loops.call_loops) + prom, kind, stripe.
    With ``fill_bp`` > 0, calls beyond the prominent summits are filled in up to n (``fill``)."""
    core = select(summits(clr, floor_q), n, min_prom, merge_bp, min_off)
    out = fill(core, top_pixels(clr), n, fill_bp, min_off, fill_cap) if fill_bp > 0 else core.assign(kind="peak")
    out = out.reset_index(drop=True)
    out["stripe"] = anchor_load(out)  # calls sharing the busier anchor (+-1): >= 5 = the call tiles a stripe
    return out[["bin1", "bin2", "value", "chrom", "pos1", "pos2", "prom", "kind", "stripe"]]


def values_near(clr, loops: pd.DataFrame, w: int = 2, field: str = "count") -> np.ndarray:
    r"""max of a cooler within +-w anchors of each loop's (bin1, bin2), on both axes (0 if no pixel). Comparing pools
    by the value at one exact pixel makes the comparison depend on where each pool's summit falls inside a ~30 kb
    DeepLoop2 peak (and on any residual one-anchor offset between checkpoints)."""
    out = np.zeros(len(loops))
    pos = np.arange(len(loops))
    for c, idx in loops.groupby("chrom").indices.items():
        px = clr.pixels().fetch(c)[["bin1_id", "bin2_id", field]]
        key = px["bin1_id"].to_numpy().astype(np.int64) * (1 << 24) + px["bin2_id"].to_numpy()
        o = np.argsort(key)
        sk, sv = key[o], px[field].to_numpy()[o]
        b1 = loops["bin1"].to_numpy()[idx].astype(np.int64)
        b2 = loops["bin2"].to_numpy()[idx].astype(np.int64)
        best = np.zeros(len(idx))
        for di in range(-w, w + 1):
            for dj in range(-w, w + 1):
                q = (b1 + di) * (1 << 24) + (b2 + dj)
                i = np.searchsorted(sk, q)
                i[i >= len(sk)] = 0
                hit = sk[i] == q
                best[hit] = np.maximum(best[hit], sv[i[hit]])
        out[pos[idx]] = best
    return out


def union_loops(loops: dict, thr: dict, merge_bp: int = 20_000) -> pd.DataFrame:
    r"""one row per loop across pools: calls of different pools within ``merge_bp`` on both ends are one loop, placed
    at the call that is strongest relative to its pool's threshold; ``called`` lists the pools (comma-separated)"""
    a = pd.concat([lp[["chrom", "bin1", "bin2", "pos1", "pos2", "value"]].assign(called=t,
                   r=np.log2(lp["value"] / thr[t])) for t, lp in loops.items()], ignore_index=True)
    a = a.sort_values("r", ascending=False).reset_index(drop=True)
    rep = np.full(len(a), -1)
    grid = {}
    for i, (c, p1, p2) in enumerate(zip(a["chrom"].to_numpy(), a["pos1"].to_numpy(), a["pos2"].to_numpy())):
        g1, g2 = p1 // max(merge_bp, 1), p2 // max(merge_bp, 1)
        for x in (g1 - 1, g1, g1 + 1):
            for y in (g2 - 1, g2, g2 + 1):
                for j, q1, q2 in grid.get((c, x, y), ()):
                    if abs(q1 - p1) <= merge_bp and abs(q2 - p2) <= merge_bp:
                        rep[i] = j
                        break
                if rep[i] >= 0:
                    break
            if rep[i] >= 0:
                break
        if rep[i] < 0:
            rep[i] = i
            grid.setdefault((c, g1, g2), []).append((i, p1, p2))
    a["rep"] = rep
    called = a.groupby("rep")["called"].agg(lambda x: ",".join(sorted(set(x), key=list(loops).index)))
    out = a.loc[called.index, ["chrom", "bin1", "bin2", "pos1", "pos2"]].assign(called=called.to_numpy())
    return out.reset_index(drop=True)
