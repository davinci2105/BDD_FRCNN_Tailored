import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import skew, kurtosis, entropy, iqr, pearsonr, spearmanr

# 0. Preliminary Setup
os.makedirs('analysis_report', exist_ok=True)
DATASET_DIR = 'bdd_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, '100k')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')
TRAIN_JSON = os.path.join(LABELS_DIR, 'bdd100k_labels_images_train.json')
VAL_JSON = os.path.join(LABELS_DIR, 'bdd100k_labels_images_val.json')
SPLITS = ['train', 'val', 'test']

def load_bdd_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

train_data = load_bdd_annotations(TRAIN_JSON)
val_data = load_bdd_annotations(VAL_JSON)

# 1. Image-Level Analysis
def image_basic_properties():
    infos = []
    for sp in SPLITS:
        paths = glob.glob(os.path.join(IMAGES_DIR, sp, '*.jpg'))
        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                    ratio = w / h if h else 0
                    ext = os.path.splitext(p)[1].lower()
            except:
                continue
            infos.append({
                'split': sp,
                'image_path': p,
                'width': w,
                'height': h,
                'aspect_ratio': ratio,
                'color_mode': mode,
                'file_extension': ext
            })
    df = pd.DataFrame(infos)
    df.to_csv('analysis_report/image_basic_properties.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['aspect_ratio'], bins=50, kde=True)
    plt.title('Aspect Ratio Dist')
    plt.xlabel('W/H')
    plt.ylabel('Count')
    plt.savefig('analysis_report/aspect_ratio_dist.png', dpi=150)
    plt.close()
    return df

def image_statistical_props():
    stats = []
    for sp in SPLITS:
        paths = glob.glob(os.path.join(IMAGES_DIR, sp, '*.jpg'))
        for p in paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            pix = g.flatten().astype(np.float32)
            m = np.mean(pix)
            s = np.std(pix)
            sk = skew(pix, bias=False)
            kt = kurtosis(pix, bias=False)
            hist, _ = np.histogram(pix, bins=256, range=(0, 255))
            hist = hist.astype(np.float32) + 1e-8
            hist /= hist.sum()
            ent = entropy(hist, base=2)
            stats.append({
                'split': sp,
                'image_path': p,
                'mean_intensity': m,
                'std_intensity': s,
                'skewness': sk,
                'kurtosis': kt,
                'entropy': ent
            })
    df = pd.DataFrame(stats)
    df.to_csv('analysis_report/image_stat_props.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['mean_intensity'], bins=50, kde=True)
    plt.title('Mean Intensity Dist')
    plt.xlabel('Mean (0-255)')
    plt.ylabel('Count')
    plt.savefig('analysis_report/mean_intensity_dist.png', dpi=150)
    plt.close()
    return df

def image_texture_analysis():
    infos = []
    for sp in SPLITS:
        paths = glob.glob(os.path.join(IMAGES_DIR, sp, '*.jpg'))
        for p in paths:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            edges = cv2.Canny(g, 100, 200)
            ed = np.sum(edges > 0) / (g.shape[0] * g.shape[1])
            lap = cv2.Laplacian(g, cv2.CV_64F)
            blur = lap.var()
            infos.append({
                'split': sp,
                'image_path': p,
                'edge_density': ed,
                'blur_score': blur
            })
    df = pd.DataFrame(infos)
    df.to_csv('analysis_report/image_texture_props.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['blur_score'], bins=50, kde=True)
    plt.title('Blur Score Dist')
    plt.xlabel('Laplacian Var')
    plt.ylabel('Count')
    plt.savefig('analysis_report/blur_score_dist.png', dpi=150)
    plt.close()
    return df

def object_distribution_in_image(tr_data, v_data):
    rows = []
    for sp, data in [('train', tr_data), ('val', v_data)]:
        for itm in data:
            fn = itm['name']
            ip = os.path.join(IMAGES_DIR, sp, fn)
            if not os.path.exists(ip):
                continue
            try:
                with Image.open(ip) as im:
                    iw, ih = im.size
                    ia = iw * ih
            except:
                continue
            total_area = 0
            if 'labels' in itm:
                for lb in itm['labels']:
                    if 'box2d' in lb:
                        x1 = lb['box2d']['x1']
                        y1 = lb['box2d']['y1']
                        x2 = lb['box2d']['x2']
                        y2 = lb['box2d']['y2']
                        total_area += (x2 - x1) * (y2 - y1)
            ratio = total_area / ia if ia else 0
            rows.append({
                'split': sp,
                'image_path': ip,
                'object_area_sum': total_area,
                'object_to_bg_ratio': ratio
            })
    df = pd.DataFrame(rows)
    df.to_csv('analysis_report/object_dist_image.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['object_to_bg_ratio'], bins=50, kde=True)
    plt.title('Obj/Bg Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Count')
    plt.savefig('analysis_report/obj_bg_ratio.png', dpi=150)
    plt.close()
    return df

# 2. BBox-Level Analysis
def get_bbox_geometry(item, sp):
    res = []
    fn = item['name']
    ip = os.path.join(IMAGES_DIR, sp, fn)
    if not os.path.exists(ip):
        return res
    try:
        with Image.open(ip) as im:
            iw, ih = im.size
            ia = iw * ih
    except:
        return res
    if 'labels' in item:
        for lb in item['labels']:
            if 'box2d' in lb:
                x1 = lb['box2d']['x1']
                y1 = lb['box2d']['y1']
                x2 = lb['box2d']['x2']
                y2 = lb['box2d']['y2']
                w = x2 - x1
                h = y2 - y1
                ba = w * h
                ar = w / h if h else 0
                ratio = ba / ia if ia else 0
                res.append({
                    'split': sp,
                    'image_path': ip,
                    'bbox_width': w,
                    'bbox_height': h,
                    'bbox_area': ba,
                    'bbox_aspect_ratio': ar,
                    'obj_img_ratio': ratio
                })
    return res

def bbox_geometry_analysis(tr_data, v_data):
    info = []
    for it in tr_data:
        info.extend(get_bbox_geometry(it, 'train'))
    for it in v_data:
        info.extend(get_bbox_geometry(it, 'val'))
    df = pd.DataFrame(info)
    df.to_csv('analysis_report/bbox_geometry.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['bbox_area'], bins=50, kde=True)
    plt.title('BBox Area Dist')
    plt.xlabel('Area')
    plt.ylabel('Count')
    plt.savefig('analysis_report/bbox_area_dist.png', dpi=150)
    plt.close()
    return df

def bbox_stats(df):
    d = {
        'mean_width': df['bbox_width'].mean(),
        'std_width': df['bbox_width'].std(),
        'mean_height': df['bbox_height'].mean(),
        'std_height': df['bbox_height'].std(),
        'mean_area': df['bbox_area'].mean(),
        'std_area': df['bbox_area'].std(),
        'var_aspect_ratio': df['bbox_aspect_ratio'].var()
    }
    pd.DataFrame([d]).to_csv('analysis_report/bbox_stats_summary.csv', index=False)
    c = df.groupby('image_path').size().reset_index(name='bbox_count')
    c.to_csv('analysis_report/bbox_density.csv', index=False)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='split', y='bbox_area', data=df)
    plt.title('BBox Area Boxplot')
    plt.savefig('analysis_report/bbox_area_boxplot.png', dpi=150)
    plt.close()

def iou_distribution(tr_data, v_data):
    def iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        uni = a1 + a2 - inter
        return inter / uni if uni else 0
    data = []
    for sp, dt in [('train', tr_data), ('val', v_data)]:
        for it in dt:
            if 'labels' not in it:
                continue
            boxes = []
            for lb in it['labels']:
                if 'box2d' in lb:
                    x1 = lb['box2d']['x1']
                    y1 = lb['box2d']['y1']
                    x2 = lb['box2d']['x2']
                    y2 = lb['box2d']['y2']
                    boxes.append((x1, y1, x2, y2))
            for (b1, b2) in combinations(boxes, 2):
                data.append({
                    'split': sp,
                    'image_id': it['name'],
                    'iou': iou(b1, b2)
                })
    df = pd.DataFrame(data)
    df.to_csv('analysis_report/iou_distribution.csv', index=False)
    if len(df):
        plt.figure(figsize=(6, 4))
        sns.histplot(df['iou'], bins=50, kde=True)
        plt.title('IoU Dist')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.savefig('analysis_report/iou_dist.png', dpi=150)
        plt.close()

# 3. Label & Class-Level
def class_distribution(tr_data, v_data):
    c = Counter()
    for sp, dt in [('train', tr_data), ('val', v_data)]:
        for it in dt:
            if 'labels' not in it:
                continue
            for lb in it['labels']:
                cls = lb.get('category', 'unknown')
                c[cls] += 1
    df = pd.DataFrame(list(c.items()), columns=['class', 'count'])
    df.to_csv('analysis_report/class_distribution.csv', index=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='class', y='count')
    plt.xticks(rotation=90)
    plt.title('Class Freq Dist')
    plt.savefig('analysis_report/class_freq_dist.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    return df

def label_stats(tr_data, v_data, df_cls):
    freqs = df_cls['count'].values.astype(np.float32)
    p = freqs / freqs.sum()
    ent = entropy(p, base=2)
    gini = 1 - np.sum(p**2)
    lpimg = []
    for sp, dt in [('train', tr_data), ('val', v_data)]:
        for it in dt:
            lpimg.append(len(it.get('labels', [])))
    d = {
        'class_entropy': ent,
        'gini_index': gini,
        'mean_labels_per_image': np.mean(lpimg),
        'std_labels_per_image': np.std(lpimg)
    }
    pd.DataFrame([d]).to_csv('analysis_report/label_stats_summary.csv', 
                             index=False)
    cooc = defaultdict(lambda: defaultdict(int))
    for sp, dt in [('train', tr_data), ('val', v_data)]:
        for it in dt:
            if 'labels' not in it:
                continue
            cset = set(lb.get('category', 'unknown') for lb in it['labels'])
            for c1 in cset:
                for c2 in cset:
                    if c1 != c2:
                        cooc[c1][c2] += 1
    ucls = list(df_cls['class'])
    mat = pd.DataFrame(0, index=ucls, columns=ucls)
    for c1 in ucls:
        for c2 in ucls:
            mat.loc[c1, c2] = cooc[c1][c2]
    mat.to_csv('analysis_report/class_cooc.csv')

# 4. Dataset Quality
def dataset_completeness(tr_data, v_data):
    ntr = len(glob.glob(os.path.join(IMAGES_DIR, 'train', '*.jpg')))
    nval = len(glob.glob(os.path.join(IMAGES_DIR, 'val', '*.jpg')))
    nte = len(glob.glob(os.path.join(IMAGES_DIR, 'test', '*.jpg')))
    tr_ann = set([i['name'] for i in tr_data])
    val_ann = set([i['name'] for i in v_data])
    mtr = ntr - len(tr_ann)
    mval = nval - len(val_ann)
    ptr = 100.0 * mtr / ntr if ntr else 0
    pval = 100.0 * mval / nval if nval else 0
    df = pd.DataFrame([{
        'split': 'train',
        'num_images': ntr,
        'images_with_annotations': len(tr_ann),
        'missing_annotations': mtr,
        'missing_percentage': ptr
    },{
        'split': 'val',
        'num_images': nval,
        'images_with_annotations': len(val_ann),
        'missing_annotations': mval,
        'missing_percentage': pval
    },{
        'split': 'test',
        'num_images': nte,
        'images_with_annotations': 0,
        'missing_annotations': 0,
        'missing_percentage': 0
    }])
    df.to_csv('analysis_report/dataset_completeness.csv', index=False)

# 5. Statistical Tools (Outlier Detection)
def detect_outliers_iqr(df):
    q1 = df['bbox_area'].quantile(0.25)
    q3 = df['bbox_area'].quantile(0.75)
    i = q3 - q1
    lb = q1 - 1.5 * i
    ub = q3 + 1.5 * i
    o = df[(df['bbox_area'] < lb) | (df['bbox_area'] > ub)]
    o.to_csv('analysis_report/outlier_bboxes_iqr.csv', index=False)

if __name__ == "__main__":
    df_img_basic = image_basic_properties()
    df_img_stat = image_statistical_props()
    df_img_tex = image_texture_analysis()
    df_obj_dist = object_distribution_in_image(train_data, val_data)
    df_bbox_geom = bbox_geometry_analysis(train_data, val_data)
    bbox_stats(df_bbox_geom)
    iou_distribution(train_data, val_data)
    df_cls = class_distribution(train_data, val_data)
    label_stats(train_data, val_data, df_cls)
    dataset_completeness(train_data, val_data)
    detect_outliers_iqr(df_bbox_geom)
    print("Done. Check analysis_report folder.")
