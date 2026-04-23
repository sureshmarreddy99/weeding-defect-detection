import os

import cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from collections import defaultdict

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')



# ---- TUNABLE PARAMETERS ----

MAD_Z_THRESH      = 2.5

MIN_ABS_PIXEL_DEV = 0.12

MIN_GT_ROWS       = 3

IOU_THRESH        = 0.45

DOT_AREA_THRESH   = 300

ANGLE_THRESH      = 15.0

BORDER_FRAC       = 0.04

GLARE_FILL_MAX    = 0.18

GLARE_BRIGHT_FRAC = 0.30

GLARE_MIN_H       = 25

GLARE_SHAPE_IOU   = 0.50

COL_MERGE_FRAC    = 0.15

CATEGORY_COLORS = {
    'bad_weeding':    (0,   80, 255),   
    'letter_missing': (255, 180,  0),   
    'misalignment':   (220,   0, 220),  
    'glare':          (0,  255, 255),   
}

CATEGORY_LABELS = {
    'bad_weeding':    'BAD-WEED',
    'letter_missing': 'MISS-LTR',
    'misalignment':   'MISALIGN',
    'glare':          'GLARE',
}

PRIORITY = ['misalignment', 'letter_missing', 'bad_weeding', 'glare']



def preprocess_image(img_bgr, blur_ksize=121, clahe_clip=2.0,
                     clahe_grid=(8,8), min_area=120):
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    norm       = cv2.divide(gray, background, scale=255)
    clahe      = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    norm_clahe = clahe.apply(norm)
    _, otsu    = cv2.threshold(norm_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive   = cv2.adaptiveThreshold(norm_clahe, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8)
    combined   = cv2.bitwise_or(otsu, adaptive)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed     = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros_like(closed)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            mask[lbl == i] = 255
    return img_rgb, gray, norm_clahe, mask



def estimate_and_deskew(img_rgb, mask):
    H, W = mask.shape[:2]
    n, lbl_cc, st_cc, cen_cc = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area   = H * W * 0.15
    all_areas  = [int(st_cc[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                  if int(st_cc[i, cv2.CC_STAT_AREA]) < max_area]
    dominant_angle = 0.0
    if all_areas:
        area_thresh = float(np.percentile(all_areas, 60))
        large_pts   = [(float(cen_cc[i][0]), float(cen_cc[i][1]))
                       for i in range(1, n)
                       if area_thresh <= int(st_cc[i, cv2.CC_STAT_AREA]) < max_area]
        if len(large_pts) >= 4:
            large_pts.sort(key=lambda p: p[1])
            med_h   = float(np.median([int(st_cc[i, cv2.CC_STAT_HEIGHT])
                                       for i in range(1, n)
                                       if area_thresh <= int(st_cc[i, cv2.CC_STAT_AREA]) < max_area]))
            row_tol = max(50, int(med_h * 0.8))
            rows_pts = []
            for pt in large_pts:
                placed = False
                for row in rows_pts:
                    if abs(pt[1] - sum(p[1] for p in row)/len(row)) < row_tol:
                        row.append(pt); placed = True; break
                if not placed: rows_pts.append([pt])
            row_angles = []
            for row in rows_pts:
                if len(row) < 2: continue
                rs = sorted(row, key=lambda p: p[0])
                c  = np.polyfit([p[0] for p in rs], [p[1] for p in rs], 1)
                row_angles.append(float(np.degrees(np.arctan(c[0]))))
            dominant_angle = float(np.median(row_angles)) if row_angles else 0.0
        if len(large_pts) < 4 or dominant_angle == 0.0:
            lines    = cv2.HoughLinesP(mask, 1, np.pi/180, 80, minLineLength=120, maxLineGap=20)
            h_angles = []
            if lines is not None:
                for ln in lines[:, 0]:
                    a = np.degrees(np.arctan2(ln[3]-ln[1], ln[2]-ln[0]))
                    if -30 <= a <= 30: h_angles.append(a)
            dominant_angle = float(np.median(h_angles)) if h_angles else 0.0
            print(f'  (Fallback Hough: {dominant_angle:.2f}°)')
    print(f'  Dominant angle: {dominant_angle:.2f}°')
    h, w = mask.shape[:2]
    M    = cv2.getRotationMatrix2D((w//2, h//2), dominant_angle, 1.0)
    rot_rgb  = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR,  borderValue=(255,255,255))
    rot_mask = cv2.warpAffine(mask,    M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot_rgb, rot_mask, dominant_angle, M



def extract_components(rot_mask, image_id, gray_img=None,
                        min_area=30, max_area_ratio=0.15):
    H, W     = rot_mask.shape[:2]
    max_area = H * W * max_area_ratio
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(rot_mask, connectivity=8)
    components = []
    for i in range(1, n):
        x    = int(stats[i, cv2.CC_STAT_LEFT]);  y    = int(stats[i, cv2.CC_STAT_TOP])
        w    = int(stats[i, cv2.CC_STAT_WIDTH]); h    = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx   = float(centroids[i][0]);           cy   = float(centroids[i][1])
        if area < min_area or area > max_area: continue

        # Orientation
        ys_i, xs_i = np.where(labels[y:y+h, x:x+w] == i)
        pts  = np.column_stack([xs_i, ys_i]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), raw_angle = rect
        angle = float(raw_angle)
        if rw < rh: angle += 90.0
        while angle >= 90:  angle -= 180
        while angle < -90:  angle += 180

        # bright_on_fg from gray image (used for reference only)
        bright_on_fg = 0.0
        if gray_img is not None:
            rm  = (labels[y:y+h, x:x+w] == i)
            fgp = gray_img[y:y+h, x:x+w][rm]
            if len(fgp) > 0:
                bright_on_fg = float((fgp > 140).sum()) / len(fgp)

        fill_ratio   = area / max(1, w * h)
        aspect_ratio = w / max(1, h)

        components.append({
            'comp_id':     i,
            'image_id':    image_id,
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': cx,       'cy': cy,
            'pixel_count': area,
            'fill_ratio':  fill_ratio,
            'aspect_ratio': aspect_ratio,
            'angle':       angle,
            'bright_on_fg': bright_on_fg,
            'is_dot':      area < DOT_AREA_THRESH,
            'label_id':    i,
            'size_class':  None, 'row_id': None,
            'col_id':      None, 'group_id': None,
            'cell_id':     None, 'is_glare': False,
            'angle_anomaly': None,
            'iou': None, 'iou_extra': None, 'iou_missing': None,
        })
    return components, labels



def classify_by_size(components, max_clusters=4):
    if len(components) < 2:
        for c in components: c['size_class'] = 'large'
        return components
    features = np.array([[c['h'], c['pixel_count']] for c in components], dtype=np.float32)
    fs       = StandardScaler().fit_transform(features)
    best_gmm, best_bic, best_k = None, np.inf, 1
    for k in range(1, max_clusters + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=0, n_init=3).fit(fs)
            bic = gmm.bic(fs)
            if bic < best_bic: best_bic, best_gmm, best_k = bic, gmm, k
        except Exception: pass
    print(f'  GMM k={best_k}')
    cluster_labels = best_gmm.predict(fs)
    ch = defaultdict(list)
    for c, l in zip(components, cluster_labels): ch[l].append(c['h'])
    sorted_cl  = sorted(ch.keys(), key=lambda k: np.mean(ch[k]))
    size_names = ['small', 'medium', 'large', 'xlarge']
    if best_k == 1:   cmap = {sorted_cl[0]: 'large'}
    elif best_k == 2: cmap = {sorted_cl[0]: 'small', sorted_cl[1]: 'large'}
    else:             cmap = {cl: size_names[min(r, 3)] for r, cl in enumerate(sorted_cl)}
    for c, l in zip(components, cluster_labels): c['size_class'] = cmap[l]
    for cl in sorted_cl:
        print(f'  [{cmap[cl]}] mean_h={np.mean(ch[cl]):.1f}px  n={len(ch[cl])}')
    return components



def assign_rows(components, y_tol_factor=0.55):
    size_rank   = {'xlarge': 3, 'large': 2, 'medium': 1, 'small': 0}
    all_sizes   = set(c['size_class'] for c in components)
    anchor_size = max(all_sizes, key=lambda s: size_rank.get(s, 0))
    anchors     = sorted([c for c in components if c['size_class'] == anchor_size],
                          key=lambda c: c['cy'])
    others      = [c for c in components if c['size_class'] != anchor_size]
    if not anchors:
        for c in components: c['row_id'] = 0
        return components, []
    med_h   = float(np.median([c['h'] for c in anchors]))
    y_tol   = max(12, int(y_tol_factor * med_h))
    rgroups = [[anchors[0]]]
    for comp in anchors[1:]:
        if abs(comp['cy'] - np.mean([c['cy'] for c in rgroups[-1]])) <= y_tol:
            rgroups[-1].append(comp)
        else:
            rgroups.append([comp])
    rgroups.sort(key=lambda g: np.mean([c['cy'] for c in g]))
    rows = []
    for idx, group in enumerate(rgroups):
        row_cy = float(np.mean([c['cy'] for c in group]))
        rows.append({'row_id': idx, 'cy': row_cy})
        for c in group: c['row_id'] = idx
    row_centers = np.array([r['cy'] for r in rows])
    attach_tol  = max(15, int(1.2 * med_h))
    for sc in others:
        d    = np.abs(row_centers - sc['cy'])
        best = int(np.argmin(d))
        sc['row_id'] = rows[best]['row_id'] if d[best] <= attach_tol else -1
    print(f'  {len(rows)} rows found')
    return components, rows



def find_columns(components, size_group, img_width, merge_frac=COL_MERGE_FRAC):
    """
    Find natural column centers for a given size group using X-projection.
    1. Collect cx of all non-dot components in the group.
    2. Use GMM to find natural clusters (columns).
    3. Merge clusters too close together (< merge_frac * img_width).
    Returns sorted list of column center x values.
    """
    cx_vals = [c['cx'] for c in components
                if c['size_class'] in size_group
                and not c.get('is_dot')
                and c.get('row_id', -1) >= 0]
    if len(cx_vals) < 2:
        return [float(np.mean(cx_vals))] if cx_vals else []

    cx_arr = np.array(cx_vals, dtype=np.float32).reshape(-1, 1)
    best_gmm, best_bic, best_k = None, np.inf, 1
    for k in range(1, min(len(cx_vals), 8) + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=0, n_init=5).fit(cx_arr)
            bic = gmm.bic(cx_arr)
            if bic < best_bic: best_bic, best_gmm, best_k = bic, gmm, k
        except Exception: pass

    col_centers = sorted(best_gmm.means_.flatten().tolist())

    # Merge columns too close together
    merge_dist = merge_frac * img_width
    merged = [col_centers[0]]
    for cx in col_centers[1:]:
        if cx - merged[-1] < merge_dist:
            merged[-1] = (merged[-1] + cx) / 2
        else:
            merged.append(cx)
    return merged



def assign_columns(components, col_centers, size_group):
    """
    Assign col_id to each component by nearest column center.
    col_id = index into sorted col_centers.
    """
    if not col_centers: return
    centers_arr = np.array(col_centers)
    for c in components:
        if c['size_class'] not in size_group: continue
        if c.get('row_id', -1) < 0: continue
        dists    = np.abs(centers_arr - c['cx'])
        c['col_id'] = int(np.argmin(dists))



def build_grid(components, rows, img_width):
    """
    Build the full grid:
    - Large/xlarge group: main letters (HARP)
    - Small/medium group: subscript letters (INC.)
    Each cell = list of components at (row_id, col_id, size_group).
    """
    LARGE_GROUP  = ('large', 'xlarge')
    SMALL_GROUP  = ('small', 'medium')

    large_cols = find_columns(components, LARGE_GROUP, img_width)
    small_cols = find_columns(components, SMALL_GROUP, img_width)
    print(f'  Large group columns: {len(large_cols)} at x={[round(c) for c in large_cols]}')
    print(f'  Small group columns: {len(small_cols)} at x={[round(c) for c in small_cols]}')

    assign_columns(components, large_cols, LARGE_GROUP)
    assign_columns(components, small_cols, SMALL_GROUP)

    # Build cell registry: keyed by (row_id, col_id, size_group)
    cells = defaultdict(list)
    for c in components:
        if c.get('row_id', -1) < 0 or c.get('col_id') is None: continue
        sg = 'large' if c['size_class'] in LARGE_GROUP else 'small'
        c['size_group'] = sg
        cell_key = (c['row_id'], c['col_id'], sg)
        c['cell_id'] = cell_key
        cells[cell_key].append(c)

    grid_info = {
        'large_cols': large_cols,
        'small_cols': small_cols,
        'cells':      dict(cells),
        'rows':       rows,
        'img_shape':  (img_width, img_width),  # placeholder; overwritten below
    }
    print(f'  Total cells occupied: {len(cells)}')
    return grid_info



def visualize_grid(rot_rgb, grid_info, components, title):
    vis  = rot_rgb.copy()
    H, W = vis.shape[:2]
    # Draw vertical column lines for large group
    for cx in grid_info['large_cols']:
        cv2.line(vis, (int(cx), 0), (int(cx), H), (0, 200, 255), 2)
    for cx in grid_info['small_cols']:
        cv2.line(vis, (int(cx), 0), (int(cx), H), (255, 165, 0), 1)
    # Draw horizontal row lines
    for row in grid_info['rows']:
        cv2.line(vis, (0, int(row['cy'])), (W, int(row['cy'])), (200, 200, 200), 1)
    # Draw components colored by col_id
    COLORS = [(255,80,80),(80,180,255),(80,255,130),(255,200,50),
              (200,80,255),(255,130,80),(80,255,200),(180,255,80)]
    for c in components:
        if c.get('col_id') is None: continue
        col = COLORS[c['col_id'] % len(COLORS)]
        cv2.rectangle(vis, (c['x'],c['y']),(c['x']+c['w'],c['y']+c['h']), col, 2)
    plt.figure(figsize=(12,9))
    plt.imshow(vis)
    patches_l = [mpatches.Patch(color=(0,.78,1), label='Large col lines'),
                 mpatches.Patch(color=(1,.65,0), label='Small col lines')]
    plt.legend(handles=patches_l, loc='upper right', fontsize=9)
    plt.title(title); plt.axis('off'); plt.show()



def build_peer_groups(all_comps, area_tol=0.30, wh_tol=0.35, angle_tol=15.0):
    groups = []

    def update_stats(g):
        m = [c for c in g['members'] if c.get('bright_on_fg', 0) <= 0.05] or g['members']
        g['mean_px']   = float(np.median([c['pixel_count']  for c in m]))
        g['mean_w']    = float(np.median([c['w']            for c in m]))
        g['mean_h']    = float(np.median([c['h']            for c in m]))
        g['mean_ang']  = float(np.median([c['angle']        for c in m]))

    def compatible(c, g):
        da = min(abs(c['angle']-g['mean_ang']), 180-abs(c['angle']-g['mean_ang']))
        return (abs(c['pixel_count']-g['mean_px']) <= area_tol*max(1,g['mean_px']) and
                abs(c['w']-g['mean_w'])            <= wh_tol  *max(1,g['mean_w'])  and
                abs(c['h']-g['mean_h'])            <= wh_tol  *max(1,g['mean_h'])  and
                da <= angle_tol)

    def score(c, g):
        da = min(abs(c['angle']-g['mean_ang']), 180-abs(c['angle']-g['mean_ang']))
        return (abs(c['pixel_count']-g['mean_px'])/max(1,g['mean_px'])*0.4 +
                abs(c['w']-g['mean_w'])           /max(1,g['mean_w']) *0.2 +
                abs(c['h']-g['mean_h'])           /max(1,g['mean_h']) *0.2 +
                da/max(1,angle_tol)                                    *0.2)

    for comp in all_comps:
        best_gid, best_sc = None, 1e18
        for g in groups:
            if compatible(comp, g):
                s = score(comp, g)
                if s < best_sc: best_sc, best_gid = s, g['gid']
        if best_gid is None:
            gid = len(groups)
            g   = {'gid': gid, 'members': [comp]}
            update_stats(g); groups.append(g); comp['group_id'] = gid
        else:
            g = groups[best_gid]
            g['members'].append(comp); update_stats(g); comp['group_id'] = best_gid

    gt_groups = [g for g in groups if len(g['members']) >= 2]
    print(f'  {len(groups)} groups, {len(gt_groups)} with size>=2')
    return gt_groups



def build_templates(rot_mask, labels, comps, gt_groups, out_size=(80, 80)):
    templates = {}
    for g in gt_groups:
        crops = []
        for m in g['members']:
            crop    = rot_mask[m['y']:m['y']+m['h'], m['x']:m['x']+m['w']]
            resized = cv2.resize((crop>0).astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST)
            crops.append(resized)
        if crops:
            stack = np.stack(crops, axis=0)
            templates[g['gid']] = (np.median(stack, axis=0) >= 0.5).astype(np.uint8)
    return templates



def compute_iou(comp, rot_mask, templates, out_size=(80, 80)):
    gid = comp.get('group_id')
    if gid is None or gid not in templates: return None, None, None
    crop  = rot_mask[comp['y']:comp['y']+comp['h'], comp['x']:comp['x']+comp['w']]
    ent   = cv2.resize((crop>0).astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST)
    tpl   = templates[gid]
    inter = np.logical_and(ent==1, tpl==1).sum()
    union = np.logical_or( ent==1, tpl==1).sum()
    ex    = np.logical_and(ent==1, tpl==0).sum() / max(1, ent.sum())
    mis   = np.logical_and(ent==0, tpl==1).sum() / max(1, tpl.sum())
    return float(inter/max(1,union)), float(ex), float(mis)



def detect_glare_from_mask(components, gray_img,
                            bright_frac_thresh=GLARE_BRIGHT_FRAC,
                            fill_max=GLARE_FILL_MAX,
                            min_h=GLARE_MIN_H,
                            shape_iou_gate=GLARE_SHAPE_IOU):
    """
    Glare detection using the mask image as primary signal:

    A component in the mask is glare if ALL of:
      1. h >= min_h  (not a tiny dot)
      2. bright_on_fg > bright_frac_thresh  (fg pixels are bright in gray = reflective)
      3. fill_ratio < fill_max  (irregular blob, not a solid letter shape)
      4. IoU vs peer template <= shape_iou_gate  (doesn't match any known letter shape)

    Strong glare override: bright_on_fg > 0.85 regardless of fill/IoU.
    """
    glare_ids = set()
    for c in components:
        c['is_glare'] = False
        if c.get('is_dot') or c['h'] < min_h: continue

        bf  = c.get('bright_on_fg', 0.0)
        iou = c.get('iou')

        # Strong glare: very high brightness → override
        if bf > 0.85:
            # Still respect IoU gate: if shape matches well, it is a shiny letter
            if iou is not None and iou > shape_iou_gate:
                continue
            c['is_glare'] = True
            glare_ids.add(c['comp_id'])
            continue

        # Normal glare: must pass all three checks
        if bf < bright_frac_thresh:              continue
        if c['fill_ratio'] >= fill_max:          continue  # solid shape = real letter
        if iou is not None and iou > shape_iou_gate: continue  # known letter shape

        c['is_glare'] = True
        glare_ids.add(c['comp_id'])

    print(f'  Glare components: {len(glare_ids)}')
    for c in components:
        if c['is_glare']:
            print(f'    comp_id={c["comp_id"]} cx={c["cx"]:.0f} cy={c["cy"]:.0f} '
                  f'bright={c["bright_on_fg"]:.3f} fill={c["fill_ratio"]:.3f} iou={c.get("iou")}')
    return components, glare_ids



def detect_angle_anomalies(components, gt_groups,
                            angle_thresh=ANGLE_THRESH):
    angle_flags = {}
    for g in gt_groups:
        clean = [c for c in g['members']
                 if not c.get('is_dot') and not c.get('is_glare')
                 and c.get('row_id', -1) >= 0]
        if len(clean) < 3: continue
        baseline = float(np.median([c['angle'] for c in clean]))
        for c in clean:
            diff = abs(c['angle'] - baseline)
            diff = min(diff, 180 - diff)
            if diff > angle_thresh:
                angle_flags[c['comp_id']] = (
                    f"angle={c['angle']:.1f}° baseline={baseline:.1f}° "
                    f"diff={diff:.1f}° group={g['gid']}"
                )
    for c in components:
        c['angle_anomaly'] = angle_flags.get(c['comp_id'])
    print(f'  Angle anomalies: {len(angle_flags)}')
    return components, angle_flags



def is_border_component(comp, img_h, img_w, border_frac=BORDER_FRAC):
    """
    True only if the component CENTROID is near the border.
    Using bounding box edges incorrectly excludes H/P letters that naturally
    start at x=0 (image crops the full letter) but whose center is well inside.
    """
    bx = int(img_w * border_frac)
    by = int(img_h * border_frac)
    return (comp['cx'] <= bx or comp['cy'] <= by or
            comp['cx'] >= img_w - bx or
            comp['cy'] >= img_h - by)



def compute_persp_scales(rows, cells, cap=1.5):
    """
    Height-based perspective normalization.

    Uses median letter HEIGHT per row (not Y position) to compute scale.
    Rationale: cy-based scaling uses median_cy/row_cy which explodes for
    rows near the image top (cy ~ 100 → scale=10x), completely masking
    real pixel count anomalies by inflating them to appear "normal".

    Height-based scale: median_h / row_median_h
    - A row with taller letters (farther in perspective) gets scale <1
    - A row with shorter letters (closer) gets scale >1
    - Anomalous pixel counts are NOT absorbed by the scale
    - The cap prevents outlier rows from producing extreme multipliers
    """
    if not rows: return {r['row_id']: 1.0 for r in rows}

    # Collect median large-letter height per row
    from collections import defaultdict
    row_heights = defaultdict(list)
    for (row_id, col_id, sg), cell_comps in cells.items():
        if sg != 'large': continue
        for c in cell_comps:
            if not c.get('is_glare') and not c.get('is_dot'):
                row_heights[row_id].append(c['h'])

    row_med_h = {}
    for r in rows:
        hs = row_heights.get(r['row_id'], [])
        if hs:
            row_med_h[r['row_id']] = float(np.median(hs))

    if not row_med_h:
        return {r['row_id']: 1.0 for r in rows}

    all_h = list(row_med_h.values())
    median_h = float(np.median(all_h))

    scales = {}
    for r in rows:
        row_h = row_med_h.get(r['row_id'], median_h)
        raw_scale = median_h / max(1.0, row_h)
        scales[r['row_id']] = float(np.clip(raw_scale, 1.0/cap, cap))
    return scales



def build_cell_ground_truth(grid_info, img_shape,
                             mad_z_thresh=MAD_Z_THRESH,
                             min_abs_dev=MIN_ABS_PIXEL_DEV,
                             min_gt_rows=MIN_GT_ROWS):
    """
    Robust column GT — fixes over v10:
    1. Border check uses CENTROID not bounding box edge
       (fixes H column being entirely excluded because x=0).
    2. Perspective scale capped at 1.8x (fixes explosion from cy=50 fragment rows).
    3. fill_ratio GT per column — used for border letters where pixel count
       and IoU are unreliable (cropped shape).
    4. Count consistency + MIN_GT_ROWS gates from v10 retained.
    """
    cells    = grid_info['cells']
    img_h, img_w = img_shape[:2]
    persp_scale  = compute_persp_scales(grid_info['rows'], cells)

    by_col = defaultdict(list)
    for (row_id, col_id, sg), cell_comps in cells.items():
        by_col[(col_id, sg)].append((row_id, cell_comps))

    col_gt = {}
    for (col_id, sg), row_entries in by_col.items():
        px_per_row    = []
        fill_per_row  = []
        count_per_row = []
        row_ids_used  = []

        for row_id, cell_comps in row_entries:
            letter_comps = [
                c for c in cell_comps
                if not c.get('is_dot')
                and not c.get('is_glare')
                and not is_border_component(c, img_h, img_w)
            ]
            # For fill GT also include border comps (fill is not affected by crop extent)
            all_letter = [c for c in cell_comps
                          if not c.get('is_dot') and not c.get('is_glare')]
            if not all_letter: continue
            # Pixel GT: non-border only
            if letter_comps:
                raw_px  = sum(c['pixel_count'] for c in letter_comps)
                scale   = persp_scale.get(row_id, 1.0)
                px_per_row.append(raw_px * scale)
                row_ids_used.append(row_id)
                count_per_row.append(len(letter_comps))
            # Fill GT: use all letters (border OK)
            mean_fill = float(np.mean([c['fill_ratio'] for c in all_letter]))
            fill_per_row.append((row_id, mean_fill))

        # Fill GT: median fill across all rows (border-insensitive)
        gt_fill    = float(np.median([f for _, f in fill_per_row])) if fill_per_row else None
        fill_mad   = float(np.median(np.abs(
            np.array([f for _, f in fill_per_row]) - gt_fill))) if fill_per_row else 0.0
        fill_mad   = max(fill_mad, (gt_fill or 0.01) * 0.05)

        # Pixel GT: needs count consistency + MIN_GT_ROWS
        gt_px = gt_mad = None
        n_clean = 0
        row_clean = []
        mode_count = None
        if len(px_per_row) >= 2:
            px_arr  = np.array(px_per_row, dtype=float)
            cnt_arr = np.array(count_per_row, dtype=float)
            mode_count  = float(np.bincount(count_per_row).argmax())
            consistent  = cnt_arr == mode_count
            px_clean    = px_arr[consistent]
            row_clean   = [r for r, ok in zip(row_ids_used, consistent) if ok]
            if len(px_clean) >= min_gt_rows:
                gt_px  = float(np.median(px_clean))
                gt_mad = float(np.median(np.abs(px_clean - gt_px)))
                gt_mad = max(gt_mad, gt_px * 0.05)
                n_clean = len(px_clean)
            else:
                mode_count = None

        col_gt[(col_id, sg)] = {
            'gt_px':        gt_px,
            'gt_mad':       gt_mad,
            'gt_count':     mode_count,
            'gt_fill':      gt_fill,
            'fill_mad':     fill_mad,
            'mad_z_thresh': mad_z_thresh,
            'min_abs_dev':  min_abs_dev,
            'n_clean':      n_clean,
            'persp_scale':  persp_scale,
            'row_ids_gt':   row_clean,
        }

    print(f'  Column GT: {len(col_gt)} combinations')
    for k, v in sorted(col_gt.items()):
        px_str   = f'gt_px={v["gt_px"]:.0f} MAD={v["gt_mad"]:.0f}' if v['gt_px'] else 'gt_px=N/A'
        fill_str = f'{v["gt_fill"]:.3f}' if v['gt_fill'] is not None else 'N/A'
        cnt_str  = f'{v["gt_count"]:.0f}' if v['gt_count'] is not None else 'N/A'
        print(f'  col={k[0]} [{k[1]}]: {px_str} fill={fill_str} '
              f'count={cnt_str} (n={v["n_clean"]} rows: {v["row_ids_gt"]})')
    return col_gt



def detect_empty_cells(grid_info, col_gt, rows):
    """
    Find (row_id, col_id, size_group) combinations that are in col_gt
    (i.e. the column exists in most rows) but have NO component in this row.
    These are definitively missing letters.
    """
    cells       = grid_info['cells']
    empty_cells = []
    occupied    = set(cells.keys())
    all_row_ids = set(r['row_id'] for r in rows)

    for (col_id, sg), gt in col_gt.items():
        if gt['gt_count'] is None or gt['gt_count'] < 0.5: continue
        for row_id in all_row_ids:
            cell_key = (row_id, col_id, sg)
            if cell_key not in occupied:
                row_has_comps = any(
                    (row_id, cid, sg2) in occupied
                    for (cid, sg2) in col_gt.keys()
                    if (row_id, cid, sg2) in occupied
                )
                if row_has_comps:
                    empty_cells.append({
                        'row_id':     row_id,
                        'col_id':     col_id,
                        'size_group': sg,
                        'cell_key':   cell_key,
                        'gt_px':      gt['gt_px'] or 0,
                    })

    print(f'  Empty cells (missing letters): {len(empty_cells)}')
    for ec in empty_cells:
        print(f'    Row {ec["row_id"]} Col {ec["col_id"]} [{ec["size_group"]}] '
              f'— expected px≈{ec["gt_px"]:.0f}')
    return empty_cells



def score_cells(grid_info, col_gt, empty_cells, angle_flags, glare_ids,
                iou_thresh=IOU_THRESH):
    cells        = grid_info['cells']
    cell_results = {}
    img_shape    = grid_info.get('img_shape', (9999, 9999))
    img_h        = img_shape[0]
    img_w        = img_shape[1]

    # --- Score occupied cells ---
    for cell_key, cell_comps in cells.items():
        row_id, col_id, sg = cell_key
        gt = col_gt.get((col_id, sg))
        flags = []

        letter_comps = [c for c in cell_comps
                        if not c.get('is_dot') and not c.get('is_glare')]
        cell_px      = sum(c['pixel_count'] for c in letter_comps)

        # 1a. Bad Weeding: pixel count (non-border letters only)
        non_border = [c for c in letter_comps
                      if not is_border_component(c, img_h, img_w)]
        if gt is not None and gt['gt_px'] is not None and non_border:
            cell_px_nb = sum(c['pixel_count'] for c in non_border)
            scale    = gt['persp_scale'].get(row_id, 1.0)
            norm_px  = cell_px_nb * scale
            gt_px    = gt['gt_px']
            gt_mad   = gt['gt_mad']
            z_score  = abs(norm_px - gt_px) / max(1.0, gt_mad * 1.4826)
            abs_dev  = abs(norm_px - gt_px) / max(1, gt_px)
            if z_score > gt['mad_z_thresh'] and abs_dev > gt['min_abs_dev']:
                if norm_px > gt_px:
                    flags.append({'category': 'bad_weeding', 'type': 'extra_material',
                                  'detail': f'px={cell_px_nb} GT={gt_px:.0f} z={z_score:.1f} (+{abs_dev*100:.1f}%)'})
                else:
                    flags.append({'category': 'bad_weeding', 'type': 'missing_material',
                                  'detail': f'px={cell_px_nb} GT={gt_px:.0f} z={z_score:.1f} ({abs_dev*100:.1f}%)'})

        # 1b. Bad Weeding: fill_ratio for border letters
        border_comps = [c for c in letter_comps
                        if is_border_component(c, img_h, img_w)]
        if gt is not None and gt.get('gt_fill') is not None and border_comps:
            mean_fill = float(np.mean([c['fill_ratio'] for c in border_comps]))
            fill_z    = abs(mean_fill - gt['gt_fill']) / max(1e-4, gt['fill_mad'] * 1.4826)
            fill_dev  = abs(mean_fill - gt['gt_fill']) / max(0.001, gt['gt_fill'])
            if fill_z > gt['mad_z_thresh'] and fill_dev > 0.12:
                bw_type = 'extra_material' if mean_fill > gt['gt_fill'] else 'missing_material'
                existing_bw_pre = {f['type'] for f in flags if f['category'] == 'bad_weeding'}
                if bw_type not in existing_bw_pre:
                    flags.append({'category': 'bad_weeding', 'type': bw_type,
                                  'detail': f'fill={mean_fill:.3f} GT_fill={gt["gt_fill"]:.3f} '
                                            f'z={fill_z:.1f} (border letter)'})

        # 2. Letter Missing: fewer letters than GT count
        if gt is not None and gt['gt_count'] is not None and gt['gt_count'] > 0:
            if len(letter_comps) < gt['gt_count'] - 0.5:
                flags.append({'category': 'letter_missing', 'type': 'missing_letter',
                              'detail': f'count={len(letter_comps)} GT={gt["gt_count"]:.0f}'})

        # 3. Misalignment
        for comp in letter_comps:
            if comp.get('angle_anomaly'):
                flags.append({'category': 'misalignment', 'type': 'angle_anomaly',
                              'detail': comp['angle_anomaly'],
                              'comp_id': comp['comp_id']})

        # 4. Bad Weeding: IoU shape
        existing_bw = {f['type'] for f in flags if f['category'] == 'bad_weeding'}
        iou_worst   = {}
        for comp in letter_comps:
            iou = comp.get('iou'); ex = comp.get('iou_extra'); mis = comp.get('iou_missing')
            if iou is None or iou >= iou_thresh: continue
            t = ('extra_material'   if ex  is not None and ex  > 0.20 else
                 'missing_material' if mis is not None and mis > 0.20 else 'shape_anomaly')
            if t not in iou_worst or iou < iou_worst[t][0]:
                iou_worst[t] = (iou, comp['comp_id'])
        for t, (iou_val, cid) in iou_worst.items():
            if t not in existing_bw:
                flags.append({'category': 'bad_weeding', 'type': t,
                              'detail': f'IoU={iou_val:.2f}', 'comp_id': cid})

        # 5. Glare — only if no bad_weeding present
        existing_cats = {f['category'] for f in flags}
        if 'bad_weeding' not in existing_cats:
            for comp in cell_comps:
                if comp['comp_id'] in glare_ids:
                    flags.append({'category': 'glare', 'type': 'glare',
                                  'detail': f'bright={comp["bright_on_fg"]:.3f}',
                                  'comp_id': comp['comp_id']})

        # Deduplicate per (category, type)
        seen, deduped = set(), []
        for f in flags:
            k = (f['category'], f['type'])
            if k not in seen: deduped.append(f); seen.add(k)

        cell_results[cell_key] = {
            'cell_key':     cell_key,
            'comps':        cell_comps,
            'letter_comps': letter_comps,
            'cell_px':      cell_px,
            'defect_flags': deduped,
            'is_defective': len(deduped) > 0,
            'categories':   list({f['category'] for f in deduped}),
            'x1': min(c['x'] for c in cell_comps),
            'y1': min(c['y'] for c in cell_comps),
            'x2': max(c['x']+c['w'] for c in cell_comps),
            'y2': max(c['y']+c['h'] for c in cell_comps),
        }

    # --- Score empty cells ---
    for ec in empty_cells:
        ck = ec['cell_key']
        row_id, col_id, sg = ck
        col_centers = (grid_info['large_cols'] if sg == 'large'
                       else grid_info['small_cols'])
        col_cx  = col_centers[col_id] if col_id < len(col_centers) else 0
        row_cy  = next((r['cy'] for r in grid_info['rows'] if r['row_id'] == row_id), 0)
        half_w  = int((ec['gt_px'] ** 0.5)) // 2 + 50 if ec['gt_px'] > 0 else 100
        half_h  = half_w
        cell_results[ck] = {
            'cell_key':     ck,
            'comps':        [],
            'letter_comps': [],
            'cell_px':      0,
            'defect_flags': [{'category': 'letter_missing', 'type': 'empty_cell',
                               'detail': f'col={col_id} expected px≈{ec["gt_px"]:.0f}'}],
            'is_defective': True,
            'categories':   ['letter_missing'],
            'x1': int(col_cx - half_w), 'y1': int(row_cy - half_h),
            'x2': int(col_cx + half_w), 'y2': int(row_cy + half_h),
            'is_empty': True,
        }

    defective  = [v for v in cell_results.values() if v['is_defective']]
    cat_counts = defaultdict(int)
    for d in defective:
        for f in d['defect_flags']: cat_counts[f['category']] += 1
    print(f'  {len(defective)}/{len(cell_results)} cells defective')
    for cat, cnt in sorted(cat_counts.items()):
        print(f'    [{cat}]: {cnt}')
    return cell_results



def draw_grid_output(rot_rgb, cell_results, grid_info, show_ok=True,
                     border_margin=0.005, draw_grid_lines=True):
    vis  = rot_rgb.copy()
    H, W = vis.shape[:2]
    bx   = int(W * border_margin); by = int(H * border_margin)

    # Draw grid lines (subtle)
    if draw_grid_lines:
        for cx in grid_info['large_cols']:
            cv2.line(vis, (int(cx),0),(int(cx),H), (80,80,80), 1)
        for cx in grid_info['small_cols']:
            cv2.line(vis, (int(cx),0),(int(cx),H), (60,60,60), 1)
        for row in grid_info['rows']:
            cv2.line(vis, (0,int(row['cy'])),(W,int(row['cy'])), (60,60,60), 1)

    for ck, res in cell_results.items():
        x1, y1, x2, y2 = res['x1'], res['y1'], res['x2'], res['y2']
        # Clamp to image
        x1 = max(bx, x1); y1 = max(by, y1)
        x2 = min(W-bx, x2); y2 = min(H-by, y2)
        if x2 <= x1 or y2 <= y1: continue

        if not res['is_defective']:
            if show_ok: cv2.rectangle(vis, (x1,y1),(x2,y2), (0,200,0), 2)
            continue

        cats    = res.get('categories', [])
        primary = next((p for p in PRIORITY if p in cats), cats[0] if cats else 'bad_weeding')
        color   = CATEGORY_COLORS.get(primary, (255,0,0))
        label   = CATEGORY_LABELS.get(primary, 'DEFECT')

        # Dashed box for empty cells
        if res.get('is_empty'):
            for dash_x in range(x1, x2, 20):
                cv2.line(vis,(dash_x,y1),(min(dash_x+10,x2),y1),color,3)
                cv2.line(vis,(dash_x,y2),(min(dash_x+10,x2),y2),color,3)
            for dash_y in range(y1, y2, 20):
                cv2.line(vis,(x1,dash_y),(x1,min(dash_y+10,y2)),color,3)
                cv2.line(vis,(x2,dash_y),(x2,min(dash_y+10,y2)),color,3)
        else:
            cv2.rectangle(vis, (x1,y1),(x2,y2), color, 4)

        cv2.putText(vis, label, (x1, max(30,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Per-component flags
        flagged = {f.get('comp_id'): f['category']
                   for f in res['defect_flags'] if 'comp_id' in f}
        for comp in res.get('comps', []):
            if comp['comp_id'] in flagged:
                fc = CATEGORY_COLORS.get(flagged[comp['comp_id']], (255,0,0))
                cv2.rectangle(vis,(comp['x'],comp['y']),
                              (comp['x']+comp['w'],comp['y']+comp['h']),fc,2)

        detail = '+'.join(CATEGORY_LABELS.get(c,c) for c in cats)
        cv2.putText(vis, detail, (x1, min(H-10,y2+22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis



def add_legend(vis):
    items = [
        ('OK',        (0, 200, 0)),
        ('BAD-WEED',  CATEGORY_COLORS['bad_weeding']),
        ('MISS-LTR',  CATEGORY_COLORS['letter_missing']),
        ('MISALIGN',  CATEGORY_COLORS['misalignment']),
        ('GLARE',     CATEGORY_COLORS['glare']),
    ]
    x0, y0, pad, lh = 12, 12, 6, 28
    cv2.rectangle(vis, (x0-pad, y0-pad), (x0+200, y0+len(items)*lh+pad*2), (40,40,40), -1)
    cv2.rectangle(vis, (x0-pad, y0-pad), (x0+200, y0+len(items)*lh+pad*2), (180,180,180), 1)
    for i, (lbl, col) in enumerate(items):
        ty = y0 + i*lh + lh//2
        cv2.rectangle(vis, (x0, ty-8), (x0+20, ty+8), col, -1)
        cv2.putText(vis, lbl, (x0+28, ty+5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230,230,230), 1)
    return vis



def plot_grid_heatmap(cell_results, col_gt, rows, grid_info, title):
    all_row_ids = sorted(set(r['row_id'] for r in rows))
    all_col_sgs = sorted(set(col_gt.keys()))
    if not all_col_sgs: return
    n_rows = len(all_row_ids)
    n_cols = len(all_col_sgs)
    data   = np.full((n_rows, n_cols), np.nan)
    annot  = [[''] * n_cols for _ in range(n_rows)]
    for ci, (col_id, sg) in enumerate(all_col_sgs):
        gt = col_gt.get((col_id, sg))
        if gt is None or gt['gt_px'] is None: continue
        for ri, row_id in enumerate(all_row_ids):
            ck = (row_id, col_id, sg)
            if ck in cell_results:
                res     = cell_results[ck]
                px      = res['cell_px']
                scale   = gt['persp_scale'].get(row_id, 1.0)
                norm_px = px * scale
                dev     = (norm_px - gt['gt_px']) / max(1, gt['gt_px'])
                data[ri, ci] = dev
                cats  = res.get('categories', [])
                label = ('+'.join(CATEGORY_LABELS.get(c, c) for c in cats)
                         if res['is_defective'] else f'{dev*100:+.0f}%')
                annot[ri][ci] = label
            else:
                data[ri, ci] = -1.5
                annot[ri][ci] = 'EMPTY'
    fig, ax = plt.subplots(figsize=(max(8, 2*n_cols), max(4, 1.5*n_rows)))
    im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'C{cid}[{sg[0].upper()}]' for cid, sg in all_col_sgs], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f'R{rid}' for rid in all_row_ids], fontsize=9)
    for ri in range(n_rows):
        for ci in range(n_cols):
            ax.text(ci, ri, annot[ri][ci], ha='center', va='center',
                    fontsize=7, color='black')
    plt.colorbar(im, ax=ax, label='Pixel deviation from GT')
    ax.set_title(f'{title}\nGreen=OK  Red=extra material  Dark=missing/empty', fontsize=10)
    plt.tight_layout(); plt.show()



def show_defect_crops(rot_rgb, cell_results, title, pad=50):
    H, W      = rot_rgb.shape[:2]
    bx, by    = int(W*0.005), int(H*0.005)
    defective = [r for r in cell_results.values()
                 if r['is_defective'] and
                 r['x1']>bx and r['y1']>by and r['x2']<W-bx and r['y2']<H-by]
    if not defective:
        print(f'{title}: No defects outside border.'); return
    n    = len(defective)
    cols = min(3, n)
    rows_g = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_g, cols, figsize=(6*cols, 5*rows_g))
    flat = np.array(axes).flatten().tolist()
    for res, ax in zip(defective, flat):
        x1 = max(0,res['x1']-pad); y1 = max(0,res['y1']-pad)
        x2 = min(W,res['x2']+pad); y2 = min(H,res['y2']+pad)
        crop = rot_rgb[y1:y2, x1:x2].copy()
        cats    = res.get('categories', [])
        primary = next((p for p in PRIORITY if p in cats), cats[0] if cats else 'bad_weeding')
        color   = CATEGORY_COLORS.get(primary, (255,0,0))
        cv2.rectangle(crop, (res['x1']-x1,res['y1']-y1),
                      (res['x2']-x1,res['y2']-y1), color, 3)
        ax.imshow(crop)
        ck = res['cell_key']
        lines = [f"R{ck[0]} C{ck[1]} [{ck[2]}]"]
        for f in res['defect_flags']:
            lines.append(f"[{f['category']}] {f.get('detail','')}")
        ax.set_title('\n'.join(lines), fontsize=7, loc='left'); ax.axis('off')
    for ax in flat[n:]: ax.axis('off')
    plt.suptitle(f'{title} — {n} defective cell(s)', fontsize=12)
    plt.tight_layout(); plt.show()




def run_detection_pipeline(img_bgr, image_id=1):
    """
    Runs the final notebook pipeline on one image and returns the annotated RGB image plus intermediate results.
    This keeps the original notebook logic/order, but removes notebook-only plotting and hardcoded paths.
    """
    img_rgb, gray, norm_clahe, mask = preprocess_image(img_bgr)
    rot_rgb, rot_mask, angle, M = estimate_and_deskew(img_rgb, mask)

    components, labels = extract_components(rot_mask, image_id, gray_img=gray)
    components = classify_by_size(components)

    components, rows = assign_rows(components)
    grid_info = build_grid(components, rows, rot_rgb.shape[1])
    grid_info['img_shape'] = rot_rgb.shape

    gt_groups = build_peer_groups(components)
    templates = build_templates(rot_mask, labels, components, gt_groups)

    for c in components:
        c['iou'], c['iou_extra'], c['iou_missing'] = compute_iou(c, rot_mask, templates)

    components, glare_ids = detect_glare_from_mask(components, gray)
    components, angle_flags = detect_angle_anomalies(components, gt_groups)

    col_gt = build_cell_ground_truth(grid_info, rot_rgb.shape)
    empty_cells = detect_empty_cells(grid_info, col_gt, rows)
    cell_results = score_cells(grid_info, col_gt, empty_cells, angle_flags, glare_ids)

    vis_rgb = add_legend(draw_grid_output(rot_rgb, cell_results, grid_info, draw_grid_lines=False))

    return {
        'annotated_rgb': vis_rgb,
        'cell_results': cell_results,
        'components': components,
        'rows': rows,
        'grid_info': grid_info,
        'col_gt': col_gt,
        'empty_cells': empty_cells,
        'angle_flags': angle_flags,
        'glare_ids': glare_ids,
        'deskew_angle': angle,
    }


def summarize_defects(cell_results):
    defects = []
    for res in cell_results.values():
        if not res.get('is_defective'):
            continue

        row_id, col_id, size_group = res['cell_key']
        for flag in res.get('defect_flags', []):
            defects.append({
                'row_id': int(row_id),
                'col_id': int(col_id),
                'size_group': size_group,
                'category': flag.get('category'),
                'type': flag.get('type'),
                'detail': flag.get('detail', ''),
                'bbox': [
                    int(res.get('x1', 0)),
                    int(res.get('y1', 0)),
                    int(res.get('x2', 0)),
                    int(res.get('y2', 0)),
                ],
            })
    return defects


def process_image(input_path, output_path):
    """
    Flask entry point.
    Reads an uploaded image, runs the final notebook defect-detection logic,
    saves the annotated output image, and returns a summary for the HTML page.
    """
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = run_detection_pipeline(img_bgr, image_id=1)
    annotated_rgb = result['annotated_rgb']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_bgr)

    defects = summarize_defects(result['cell_results'])

    return {
        'output_path': output_path,
        'defect_count': len(defects),
        'defects': defects,
    }
