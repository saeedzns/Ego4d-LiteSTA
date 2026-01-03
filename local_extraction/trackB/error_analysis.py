#!/usr/bin/env python3
"""
Track B Error Analysis Tool

Generates comprehensive error analysis from evaluation metrics and predictions:
1. Per-noun accuracy bar charts (worst/best classes)
2. Per-verb accuracy bar charts (worst/best classes)
3. Small object failure analysis (by box area)
4. TTC error distribution histogram
5. Confusion matrix for noun predictions
6. Auto-selects failure case overlays for review

Usage:
    python local_extraction/trackB/error_analysis.py
    python local_extraction/trackB/error_analysis.py --metrics path/to/metrics.json
    python local_extraction/trackB/error_analysis.py --predictions path/to/predictions.csv
    python local_extraction/trackB/error_analysis.py --top_k 20  # Show top/bottom 20 classes
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[error_analysis] Warning: matplotlib not installed. Plots will be skipped.")

# Try to import PIL for custom overlays
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[error_analysis] Warning: PIL not installed. Custom overlays will be skipped.")


def save_plot_data(data: Dict[str, Any], filename: str, output_dir: Path) -> None:
    """Save plot data to JSON file."""
    json_path = output_dir / f"{filename}_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[error_analysis] Saved data: {json_path.name}")


# ===================== Configuration =====================

@dataclass
class AnalysisConfig:
    """Configuration for error analysis."""
    metrics_dir: Path = Path("local_extraction/runs/Track_B/metrics")
    predictions_dir: Path = Path("local_extraction/runs/Track_B/predictions")
    overlays_dir: Path = Path("local_extraction/runs/Track_B/overlays/val")
    output_dir: Path = Path("local_extraction/runs/Track_B/error_analysis")
    noun_mapping_path: Optional[Path] = Path("local_extraction/v2/org_annotations/fho_sta_val_height-540.json")
    verb_mapping_path: Optional[Path] = None  # Will try to load from annotations
    top_k: int = 15  # Number of top/bottom classes to show
    min_samples: int = 3  # Minimum samples for a class to be included
    small_object_threshold: float = 0.01  # Box area as fraction of image (1% = small)
    failure_gallery_count: int = 20  # Number of failure cases to copy


# ===================== Data Loading =====================

def load_latest_metrics(metrics_dir: Path) -> Tuple[Dict[str, Any], Path]:
    """Load the most recent metrics JSON file."""
    files = sorted(metrics_dir.glob("metrics_val_*.json"))
    # Exclude summary files
    files = [f for f in files if "_summary" not in f.name]
    if not files:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir}")
    latest = files[-1]
    print(f"[error_analysis] Loading metrics: {latest.name}")
    with latest.open("r", encoding="utf-8") as f:
        return json.load(f), latest


def load_latest_predictions(predictions_dir: Path) -> Tuple[List[Dict], Path]:
    """Load the most recent predictions CSV file."""
    files = sorted(predictions_dir.glob("predictions_val_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions files found in {predictions_dir}")
    latest = files[-1]
    print(f"[error_analysis] Loading predictions: {latest.name}")
    rows = []
    with latest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows, latest


def load_noun_names(path: Optional[Path]) -> Dict[int, str]:
    """Load noun ID to name mapping from annotations."""
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Primary: check for noun_categories list (Ego4D format)
        if "noun_categories" in data:
            noun_names = {}
            for cat in data["noun_categories"]:
                nid = cat.get("id")
                nname = cat.get("name", "")
                if nid is not None:
                    # Extract short name (first word before parentheses)
                    short_name = nname.split("_(")[0] if "_(" in nname else nname.split("(")[0].strip()
                    noun_names[int(nid)] = short_name
            if noun_names:
                print(f"[error_analysis] Loaded {len(noun_names)} noun names from noun_categories")
                return noun_names
        
        # Fallback: Try to find noun taxonomy
        if "noun_classes" in data:
            return {int(k): v for k, v in data["noun_classes"].items()}
        if "taxonomy" in data and "noun" in data["taxonomy"]:
            return {i: n for i, n in enumerate(data["taxonomy"]["noun"])}
        # Fallback: Try to extract from annotations
        noun_names = {}
        for anno in data.get("annotations", []):
            for obj in anno.get("objects", []):
                nid = obj.get("noun_category_id")
                nname = obj.get("noun_category")
                if nid is not None and nname:
                    noun_names[int(nid)] = nname
        return noun_names
    except Exception as e:
        print(f"[error_analysis] Warning: Could not load noun names: {e}")
        return {}


def load_verb_names(path: Optional[Path]) -> Dict[int, str]:
    """Load verb ID to name mapping from annotations."""
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Primary: check for verb_categories list (Ego4D format)
        if "verb_categories" in data:
            verb_names = {}
            for cat in data["verb_categories"]:
                vid = cat.get("id")
                vname = cat.get("name", "")
                if vid is not None:
                    # Extract short name (first word before parentheses)
                    short_name = vname.split("_(")[0] if "_(" in vname else vname.split("(")[0].strip()
                    verb_names[int(vid)] = short_name
            if verb_names:
                print(f"[error_analysis] Loaded {len(verb_names)} verb names from verb_categories")
                return verb_names
        
        if "verb_classes" in data:
            return {int(k): v for k, v in data["verb_classes"].items()}
        if "taxonomy" in data and "verb" in data["taxonomy"]:
            return {i: v for i, v in enumerate(data["taxonomy"]["verb"])}
        # Fallback: Try to extract from annotations
        verb_names = {}
        for anno in data.get("annotations", []):
            vid = anno.get("verb_category_id")
            vname = anno.get("verb_category")
            if vid is not None and vname:
                verb_names[int(vid)] = vname
        return verb_names
    except Exception as e:
        print(f"[error_analysis] Warning: Could not load verb names: {e}")
        return {}


# ===================== Analysis Functions =====================

def analyze_per_class_accuracy(
    stats: Dict[str, Dict],
    class_names: Dict[int, str],
    class_type: str,  # "noun" or "verb"
    top_k: int = 15,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """Analyze per-class accuracy statistics."""
    results = {
        "class_type": class_type,
        "total_classes": len(stats),
        "classes_with_min_samples": 0,
        "worst_classes": [],
        "best_classes": [],
        "zero_accuracy_classes": [],
        "perfect_accuracy_classes": [],
        "overall_accuracy": 0.0,
    }
    
    # Filter classes with minimum samples
    filtered = []
    total_correct = 0
    total_samples = 0
    
    for cid_str, data in stats.items():
        cid = int(cid_str)
        total = data.get("total", 0)
        correct = data.get("correct", 0)
        acc = data.get("accuracy", 0.0)
        
        total_samples += total
        total_correct += correct
        
        if total >= min_samples:
            name = class_names.get(cid, f"{class_type}_{cid}")
            filtered.append({
                "id": cid,
                "name": name,
                "total": total,
                "correct": correct,
                "accuracy": acc,
            })
    
    results["classes_with_min_samples"] = len(filtered)
    results["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Sort by accuracy
    sorted_by_acc = sorted(filtered, key=lambda x: (x["accuracy"], -x["total"]))
    
    # Worst classes (lowest accuracy)
    results["worst_classes"] = sorted_by_acc[:top_k]
    
    # Best classes (highest accuracy, excluding perfect with few samples)
    results["best_classes"] = sorted_by_acc[-top_k:][::-1]
    
    # Zero accuracy classes
    results["zero_accuracy_classes"] = [c for c in filtered if c["accuracy"] == 0.0]
    
    # Perfect accuracy classes
    results["perfect_accuracy_classes"] = [c for c in filtered if c["accuracy"] == 1.0]
    
    return results


def analyze_box_size_failures(
    predictions: List[Dict],
    small_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Analyze failures by bounding box size."""
    results = {
        "small_object_threshold": small_threshold,
        "total_predictions": len(predictions),
        "small_objects": {"total": 0, "correct": 0, "accuracy": 0.0},
        "medium_objects": {"total": 0, "correct": 0, "accuracy": 0.0},
        "large_objects": {"total": 0, "correct": 0, "accuracy": 0.0},
        "size_bins": [],
    }
    
    # Compute box areas from x1,y1,x2,y2 columns
    for pred in predictions:
        try:
            x1 = float(pred.get("x1", 0))
            y1 = float(pred.get("y1", 0))
            x2 = float(pred.get("x2", 0))
            y2 = float(pred.get("y2", 0))
            
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            # Normalize assuming 960x540 image
            width_norm = width / 960
            height_norm = height / 540
            area = width_norm * height_norm
            pred["_box_area"] = area
        except:
            pred["_box_area"] = None
    
    # Categorize by size
    for pred in predictions:
        area = pred.get("_box_area")
        if area is None:
            continue
        
        # Determine correctness (check if prediction matches GT)
        is_correct = False
        try:
            gt_noun = int(pred.get("gt_noun_id", pred.get("gt_noun", -1)))
            pred_noun = int(pred.get("pred_noun_id", pred.get("pred_noun", -2)))
            is_correct = (gt_noun == pred_noun)
        except:
            pass
        
        # Categorize
        if area < small_threshold:
            category = "small_objects"
        elif area < small_threshold * 5:
            category = "medium_objects"
        else:
            category = "large_objects"
        
        results[category]["total"] += 1
        if is_correct:
            results[category]["correct"] += 1
    
    # Calculate accuracies
    for cat in ["small_objects", "medium_objects", "large_objects"]:
        total = results[cat]["total"]
        correct = results[cat]["correct"]
        results[cat]["accuracy"] = correct / total if total > 0 else 0.0
    
    return results


def analyze_ttc_errors(predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze TTC prediction errors."""
    results = {
        "total_with_ttc": 0,
        "mean_error": 0.0,
        "median_error": 0.0,
        "std_error": 0.0,
        "errors": [],
        "error_bins": {},
    }
    
    errors = []
    for pred in predictions:
        try:
            # Try multiple column name formats
            gt_ttc = float(pred.get("ttc_gt_s", pred.get("gt_ttc", 0)))
            pred_ttc = float(pred.get("ttc_pred_s", pred.get("pred_ttc", 0)))
            error = abs(pred_ttc - gt_ttc)
            errors.append(error)
            pred["_ttc_error"] = error
        except:
            pass
    
    if errors:
        results["total_with_ttc"] = len(errors)
        results["mean_error"] = float(np.mean(errors))
        results["median_error"] = float(np.median(errors))
        results["std_error"] = float(np.std(errors))
        results["errors"] = errors
        
        # Bin errors
        bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, float("inf")]
        bin_labels = ["<0.1s", "0.1-0.2s", "0.2-0.3s", "0.3-0.5s", "0.5-1.0s", ">1.0s"]
        for i, label in enumerate(bin_labels):
            count = sum(1 for e in errors if bins[i] <= e < bins[i+1])
            results["error_bins"][label] = count
    
    return results


def build_confusion_matrix(
    predictions: List[Dict],
    class_names: Dict[int, str],
    top_k_confusions: int = 10,
) -> Dict[str, Any]:
    """Build noun confusion matrix and find top confusions."""
    confusion = defaultdict(lambda: defaultdict(int))
    
    for pred in predictions:
        try:
            gt_noun = int(pred.get("gt_noun_id", pred.get("gt_noun", -1)))
            pred_noun = int(pred.get("pred_noun_id", pred.get("pred_noun", -1)))
            if gt_noun >= 0 and pred_noun >= 0:
                confusion[gt_noun][pred_noun] += 1
        except:
            pass
    
    # Find top confusion pairs (excluding correct predictions)
    confusion_pairs = []
    for gt, preds in confusion.items():
        for pred, count in preds.items():
            if gt != pred and count > 0:
                gt_name = class_names.get(gt, f"noun_{gt}")
                pred_name = class_names.get(pred, f"noun_{pred}")
                confusion_pairs.append({
                    "gt_id": gt,
                    "gt_name": gt_name,
                    "pred_id": pred,
                    "pred_name": pred_name,
                    "count": count,
                })
    
    # Sort by count
    confusion_pairs.sort(key=lambda x: -x["count"])
    
    return {
        "total_confusions": sum(p["count"] for p in confusion_pairs),
        "unique_confusion_pairs": len(confusion_pairs),
        "top_confusions": confusion_pairs[:top_k_confusions],
    }


def select_failure_cases(
    predictions: List[Dict],
    overlays_dir: Path,
    output_dir: Path,
    count: int = 20,
    noun_names: Optional[Dict[int, str]] = None,
    verb_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Select and create custom overlays for worst failure cases with clear error visualization."""
    failures = []
    noun_names = noun_names or {}
    verb_names = verb_names or {}
    
    # Score each prediction by "badness"
    for pred in predictions:
        try:
            # Skip if no GT box available (can't visualize comparison)
            has_gt_box = all(pred.get(k) not in [None, "", "None"] for k in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"])
            
            # Check if incorrect
            gt_noun = int(pred.get("gt_noun_id", pred.get("gt_noun", -1)))
            pred_noun = int(pred.get("pred_noun_id", pred.get("pred_noun", -1)))
            is_noun_wrong = (gt_noun != pred_noun) and gt_noun >= 0
            
            gt_verb = int(pred.get("gt_verb_id", pred.get("gt_verb", -1)))
            pred_verb = int(pred.get("pred_verb_id", pred.get("pred_verb", -1)))
            is_verb_wrong = (gt_verb != pred_verb) and gt_verb >= 0
            
            ttc_error = pred.get("_ttc_error", 0)
            if ttc_error is None:
                ttc_error = 0
            
            # Compute badness score
            badness = 0
            if is_noun_wrong:
                badness += 2
            if is_verb_wrong:
                badness += 1
            badness += min(float(ttc_error), 1.0)  # Cap TTC contribution
            
            if badness > 0:
                pred["_badness"] = badness
                pred["_is_noun_wrong"] = is_noun_wrong
                pred["_is_verb_wrong"] = is_verb_wrong
                pred["_has_gt_box"] = has_gt_box
                pred["_gt_noun_name"] = noun_names.get(gt_noun, f"noun_{gt_noun}")
                pred["_pred_noun_name"] = noun_names.get(pred_noun, f"noun_{pred_noun}")
                pred["_gt_verb_name"] = verb_names.get(gt_verb, f"verb_{gt_verb}")
                pred["_pred_verb_name"] = verb_names.get(pred_verb, f"verb_{pred_verb}")
                failures.append(pred)
        except Exception as e:
            pass
    
    # Sort by badness, but prioritize failures WITH GT box for better visualization
    failures.sort(key=lambda x: (-int(x.get("_has_gt_box", False)), -x.get("_badness", 0)))
    
    # Select top failures
    selected = failures[:count]
    
    # Create gallery directory (overwrite old)
    gallery_dir = output_dir / "failure_gallery"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom error overlays if PIL available
    created = []
    if HAS_PIL:
        for i, pred in enumerate(selected):
            try:
                frame_path = pred.get("frame_path", "")
                if not frame_path or not Path(frame_path).exists():
                    continue
                
                # Create custom error overlay with consistent filename
                overlay_path = _create_error_overlay(
                    frame_path=Path(frame_path),
                    pred=pred,
                    output_path=gallery_dir / f"failure_{i+1:02d}.jpg",
                    rank=i + 1,
                )
                if overlay_path:
                    pred["_overlay_created"] = str(overlay_path)
                    created.append(pred)
            except Exception as e:
                print(f"[error_analysis] Warning: Could not create overlay: {e}")
        
        print(f"[error_analysis] Created {len(created)} failure overlays in {gallery_dir}")
    else:
        print(f"[error_analysis] PIL not available, skipping failure overlay creation")
    
    # Create summary text file with legend
    _create_failure_summary(selected, gallery_dir, noun_names, verb_names)
    
    return selected


def select_success_cases(
    predictions: List[Dict],
    overlays_dir: Path,
    output_dir: Path,
    count: int = 20,
    noun_names: Optional[Dict[int, str]] = None,
    verb_names: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """Select and create custom overlays for best success cases (correct predictions)."""
    successes = []
    noun_names = noun_names or {}
    verb_names = verb_names or {}
    
    # Score each prediction by "goodness" (fully correct predictions)
    for pred in predictions:
        try:
            # Check if correct
            gt_noun = int(pred.get("gt_noun_id", pred.get("gt_noun", -1)))
            pred_noun = int(pred.get("pred_noun_id", pred.get("pred_noun", -1)))
            is_noun_correct = (gt_noun == pred_noun) and gt_noun >= 0
            
            gt_verb = int(pred.get("gt_verb_id", pred.get("gt_verb", -1)))
            pred_verb = int(pred.get("pred_verb_id", pred.get("pred_verb", -1)))
            is_verb_correct = (gt_verb == pred_verb) and gt_verb >= 0
            
            ttc_error = pred.get("_ttc_error", 0)
            if ttc_error is None:
                ttc_error = 0
            
            # Check if TTC is within threshold (0.25s official)
            is_ttc_correct = float(ttc_error) <= 0.25
            
            # Compute goodness score (higher is better)
            goodness = 0
            if is_noun_correct:
                goodness += 2
            if is_verb_correct:
                goodness += 1
            if is_ttc_correct:
                goodness += 1
            # Bonus for low TTC error
            goodness += max(0, 1.0 - float(ttc_error))
            
            # Only include if at least noun is correct
            if is_noun_correct:
                pred["_goodness"] = goodness
                pred["_is_noun_correct"] = is_noun_correct
                pred["_is_verb_correct"] = is_verb_correct
                pred["_is_ttc_correct"] = is_ttc_correct
                pred["_gt_noun_name"] = noun_names.get(gt_noun, f"noun_{gt_noun}")
                pred["_pred_noun_name"] = noun_names.get(pred_noun, f"noun_{pred_noun}")
                pred["_gt_verb_name"] = verb_names.get(gt_verb, f"verb_{gt_verb}")
                pred["_pred_verb_name"] = verb_names.get(pred_verb, f"verb_{pred_verb}")
                pred["_ttc_error_s"] = float(ttc_error)
                successes.append(pred)
        except Exception as e:
            pass
    
    # Sort by goodness (best first)
    successes.sort(key=lambda x: -x.get("_goodness", 0))
    
    # Select top successes
    selected = successes[:count]
    
    # Create gallery directory (overwrite old)
    gallery_dir = output_dir / "success_gallery"
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom success overlays if PIL available
    created = []
    if HAS_PIL:
        for i, pred in enumerate(selected):
            try:
                frame_path = pred.get("frame_path", "")
                if not frame_path or not Path(frame_path).exists():
                    continue
                
                # Create custom success overlay
                overlay_path = _create_success_overlay(
                    frame_path=Path(frame_path),
                    pred=pred,
                    output_path=gallery_dir / f"success_{i+1:02d}.jpg",
                    rank=i + 1,
                )
                if overlay_path:
                    pred["_overlay_created"] = str(overlay_path)
                    created.append(pred)
            except Exception as e:
                print(f"[error_analysis] Warning: Could not create success overlay: {e}")
        
        print(f"[error_analysis] Created {len(created)} success overlays in {gallery_dir}")
    else:
        print(f"[error_analysis] PIL not available, skipping success overlay creation")
    
    # Create summary text file
    _create_success_summary(selected, gallery_dir, noun_names, verb_names)
    
    return selected


def _create_success_overlay(
    frame_path: Path,
    pred: Dict,
    output_path: Path,
    rank: int,
) -> Optional[Path]:
    """Create a custom success overlay with clear color coding and legend (same design as failures)."""
    if not HAS_PIL:
        return None
    
    try:
        # Load image
        img = Image.open(str(frame_path)).convert('RGB')
        draw = ImageDraw.Draw(img)
        W, H = img.size
        
        # Colors (RGB for PIL)
        COLOR_GT = (46, 204, 113)       # Green - Ground Truth
        COLOR_PRED_OK = (52, 152, 219)  # Blue - Correct prediction
        COLOR_SUCCESS = (46, 204, 113)  # Green - Success marker
        COLOR_PARTIAL = (255, 193, 7)   # Yellow - Partial success (noun ok, verb wrong)
        
        # Get box coordinates - support multiple formats
        def get_pred_box(p):
            if all(k in p for k in ["x1", "y1", "x2", "y2"]):
                try:
                    return [float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])]
                except:
                    pass
            bbox_str = p.get("bbox", p.get("box", ""))
            return parse_bbox_str(bbox_str)
        
        def get_gt_box(p):
            if all(k in p for k in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]):
                try:
                    return [float(p["gt_x1"]), float(p["gt_y1"]), float(p["gt_x2"]), float(p["gt_y2"])]
                except:
                    pass
            gt_bbox_str = p.get("gt_bbox", p.get("gt_box", ""))
            return parse_bbox_str(gt_bbox_str)
        
        def parse_bbox_str(s):
            if not s:
                return None
            try:
                s = str(s).replace("[", "").replace("]", "").replace(" ", "")
                parts = s.split(",")
                if len(parts) >= 4:
                    return [float(x) for x in parts[:4]]
            except:
                pass
            return None
        
        def compute_iou(box1, box2):
            """Compute IoU between two boxes [x1, y1, x2, y2]."""
            if box1 is None or box2 is None:
                return 0.0
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0
        
        pred_box = get_pred_box(pred)
        gt_box = get_gt_box(pred)
        
        # Compute IoU between pred and GT
        iou = compute_iou(pred_box, gt_box)
        
        # Get correctness info
        is_noun_correct = pred.get("_is_noun_correct", False)
        is_verb_correct = pred.get("_is_verb_correct", False)
        is_ttc_correct = pred.get("_is_ttc_correct", False)
        
        # Get names
        gt_noun_name = pred.get("_gt_noun_name", "?")
        pred_noun_name = pred.get("_pred_noun_name", "?")
        gt_verb_name = pred.get("_gt_verb_name", "?")
        pred_verb_name = pred.get("_pred_verb_name", "?")
        
        # TTC info
        gt_ttc = pred.get("gt_ttc", pred.get("ttc_gt_s", pred.get("ttc_gt", "?")))
        pred_ttc = pred.get("pred_ttc", pred.get("ttc_pred_s", pred.get("ttc_pred", "?")))
        try:
            gt_ttc_f = float(gt_ttc)
            pred_ttc_f = float(pred_ttc)
            ttc_err = abs(pred_ttc_f - gt_ttc_f)
            gt_ttc_str = f"{gt_ttc_f:.2f}s"
            pred_ttc_str = f"{pred_ttc_f:.2f}s"
        except:
            gt_ttc_str = str(gt_ttc)
            pred_ttc_str = str(pred_ttc)
            ttc_err = pred.get("_ttc_error_s", 0)
        
        # Draw ground truth box with label (GREEN)
        if gt_box:
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline=COLOR_GT, width=4)
            
            # Build GT label
            gt_label = f"GT: {gt_noun_name}, {gt_verb_name}, ttc={gt_ttc_str}"
            
            # Draw label above box with background
            label_y = max(0, y1 - 20)
            text_bbox = draw.textbbox((x1, label_y), gt_label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((x1, label_y), gt_label, fill=COLOR_GT)
        
        # Draw predicted box with label (BLUE for correct)
        if pred_box:
            x1, y1, x2, y2 = pred_box
            
            # Choose color based on full correctness
            if is_noun_correct and is_verb_correct:
                color = COLOR_PRED_OK  # Blue - fully correct
            else:
                color = COLOR_PARTIAL  # Yellow - partial (noun ok, verb wrong)
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Build prediction label with success markers
            noun_part = f"{pred_noun_name} ✓"
            verb_part = f"{pred_verb_name}"
            if is_verb_correct:
                verb_part += " ✓"
            else:
                verb_part += " ✗"
            
            pred_label = f"PRED: {noun_part}, {verb_part}, ttc={pred_ttc_str}"
            
            # Draw label below box with background
            label_y = min(H - 20, y2 + 5)
            text_bbox = draw.textbbox((x1, label_y), pred_label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((x1, label_y), pred_label, fill=color)
        
        # Draw rank, IoU, and summary in top-left corner
        goodness = pred.get("_goodness", 0)
        summary_lines = [f"✓ SUCCESS #{rank}  IoU={iou:.2f}"]
        summary_lines.append(f"Goodness: {goodness:.2f}")
        summary_lines.append(f"Noun: {pred_noun_name} ✓")
        verb_mark = "✓" if is_verb_correct else "✗"
        summary_lines.append(f"Verb: {pred_verb_name} {verb_mark}")
        ttc_mark = "✓" if is_ttc_correct else ""
        summary_lines.append(f"TTC err: {ttc_err:.3f}s {ttc_mark}")
        
        y_pos = 10
        for line in summary_lines:
            text_bbox = draw.textbbox((10, y_pos), line)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            # First line in green (SUCCESS), rest in white
            text_color = COLOR_SUCCESS if y_pos == 10 else (255, 255, 255)
            draw.text((10, y_pos), line, fill=text_color)
            y_pos += 20
        
        # Draw legend in top-right corner
        legend_items = [
            ("■ GT (green)", COLOR_GT),
            ("■ Pred OK (blue)", COLOR_PRED_OK),
            ("■ Partial OK (yellow)", COLOR_PARTIAL),
            ("✓ = Correct", COLOR_SUCCESS),
        ]
        legend_x = W - 160
        legend_y = 10
        for label, color in legend_items:
            text_bbox = draw.textbbox((legend_x, legend_y), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((legend_x, legend_y), label, fill=color)
            legend_y += 18
        
        # Save
        img.save(str(output_path), quality=95)
        return output_path
        
    except Exception as e:
        print(f"[error_analysis] Error creating success overlay: {e}")
        return None


def _create_success_summary(
    successes: List[Dict],
    output_dir: Path,
    noun_names: Dict[int, str],
    verb_names: Dict[int, str],
) -> None:
    """Create a text summary of all success cases (same format as failure summary)."""
    summary_path = output_dir / "success_summary.txt"
    
    lines = [
        "=" * 70,
        "SUCCESS CASE GALLERY - SUMMARY",
        "=" * 70,
        "",
        "Legend:",
        "  - Green box: Ground Truth (GT)",
        "  - Blue box: Correct prediction (noun + verb correct)",
        "  - Yellow box: Partial success (noun correct, verb wrong)",
        "  - ✓ = Correct prediction",
        "  - TTC threshold = 0.25s (official Ego4D)",
        "",
        "Files are named: success_01.jpg, success_02.jpg, etc.",
        "Ranked by 'goodness' score (noun +2, verb +1, TTC<0.25s +1)",
        "",
        "-" * 70,
        "",
    ]
    
    for i, s in enumerate(successes):
        goodness = s.get("_goodness", 0)
        lines.append(f"#{i+1:02d} - goodness={goodness:.2f}")
        lines.append(f"    UID: {s.get('uid', '?')}")
        lines.append(f"    Frame: {s.get('frame_path', '?')}")
        
        gt_noun = s.get("_gt_noun_name", "?")
        gt_verb = s.get("_gt_verb_name", "?")
        pred_verb = s.get("_pred_verb_name", "?")
        is_verb_correct = s.get("_is_verb_correct", False)
        is_ttc_correct = s.get("_is_ttc_correct", False)
        ttc_err = s.get("_ttc_error_s", 0)
        
        lines.append(f"    Noun: {gt_noun} ✓ (CORRECT)")
        verb_status = "CORRECT" if is_verb_correct else f"WRONG (pred: {pred_verb})"
        lines.append(f"    Verb: {gt_verb} - {verb_status}")
        ttc_status = "OK (<0.25s)" if is_ttc_correct else "EXCEEDED"
        lines.append(f"    TTC error: {ttc_err:.3f}s - {ttc_status}")
        lines.append("")
    
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"[error_analysis] Saved success summary: {summary_path}")


def _create_error_overlay(
    frame_path: Path,
    pred: Dict,
    output_path: Path,
    rank: int,
) -> Optional[Path]:
    """Create a custom error overlay with clear color coding and legend."""
    if not HAS_PIL:
        return None
    
    try:
        # Load image
        img = Image.open(str(frame_path)).convert('RGB')
        draw = ImageDraw.Draw(img)
        W, H = img.size
        
        # Colors (RGB for PIL)
        COLOR_GT = (46, 204, 113)       # Green - Ground Truth
        COLOR_PRED_OK = (52, 152, 219)  # Blue - Correct prediction
        COLOR_PRED_ERR = (231, 76, 60)  # Red - Wrong prediction
        COLOR_SAME_BOX = (255, 193, 7)  # Yellow/Gold - Same box as GT
        
        # Try to get box coordinates - support multiple formats
        def get_pred_box(p):
            # Format 1: separate x1,y1,x2,y2 columns
            if all(k in p for k in ["x1", "y1", "x2", "y2"]):
                try:
                    return [float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])]
                except:
                    pass
            # Format 2: bbox string
            bbox_str = p.get("bbox", p.get("box", ""))
            return parse_bbox_str(bbox_str)
        
        def get_gt_box(p):
            # Format 1: separate gt_x1,gt_y1,gt_x2,gt_y2 columns
            if all(k in p for k in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]):
                try:
                    return [float(p["gt_x1"]), float(p["gt_y1"]), float(p["gt_x2"]), float(p["gt_y2"])]
                except:
                    pass
            # Format 2: gt_bbox string
            gt_bbox_str = p.get("gt_bbox", p.get("gt_box", ""))
            return parse_bbox_str(gt_bbox_str)
        
        def parse_bbox_str(s):
            if not s:
                return None
            try:
                s = str(s).replace("[", "").replace("]", "").replace(" ", "")
                parts = s.split(",")
                if len(parts) >= 4:
                    return [float(x) for x in parts[:4]]
            except:
                pass
            return None
        
        def compute_iou(box1, box2):
            """Compute IoU between two boxes [x1, y1, x2, y2]."""
            if box1 is None or box2 is None:
                return 0.0
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0
        
        pred_box = get_pred_box(pred)
        gt_box = get_gt_box(pred)
        
        # Compute IoU between pred and GT
        iou = compute_iou(pred_box, gt_box)
        is_same_box = iou > 0.95  # Consider same box if IoU > 95%
        
        # Determine error type
        is_noun_wrong = pred.get("_is_noun_wrong", False)
        is_verb_wrong = pred.get("_is_verb_wrong", False)
        
        # Get names
        gt_noun_name = pred.get("_gt_noun_name", "?")
        pred_noun_name = pred.get("_pred_noun_name", "?")
        gt_verb_name = pred.get("_gt_verb_name", "?")
        pred_verb_name = pred.get("_pred_verb_name", "?")
        
        # TTC info
        gt_ttc = pred.get("gt_ttc", pred.get("ttc_gt_s", pred.get("ttc_gt", "?")))
        pred_ttc = pred.get("pred_ttc", pred.get("ttc_pred_s", pred.get("ttc_pred", "?")))
        try:
            gt_ttc_f = float(gt_ttc)
            pred_ttc_f = float(pred_ttc)
            ttc_err = abs(pred_ttc_f - gt_ttc_f)
            gt_ttc_str = f"{gt_ttc_f:.2f}s"
            pred_ttc_str = f"{pred_ttc_f:.2f}s"
        except:
            gt_ttc_str = str(gt_ttc)
            pred_ttc_str = str(pred_ttc)
            ttc_err = 0
        
        # Draw ground truth box with label (GREEN)
        if gt_box:
            x1, y1, x2, y2 = gt_box
            draw.rectangle([x1, y1, x2, y2], outline=COLOR_GT, width=4)
            
            # Build GT label
            gt_label = f"GT: {gt_noun_name}, {gt_verb_name}, ttc={gt_ttc_str}"
            
            # Draw label above box with background
            label_y = max(0, y1 - 20)
            text_bbox = draw.textbbox((x1, label_y), gt_label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((x1, label_y), gt_label, fill=COLOR_GT)
        
        # Draw predicted box with label
        if pred_box:
            x1, y1, x2, y2 = pred_box
            is_wrong = is_noun_wrong or is_verb_wrong
            
            # Choose color based on correctness and whether same box
            if is_same_box:
                color = COLOR_SAME_BOX  # Yellow - same box but wrong classification
            elif is_wrong:
                color = COLOR_PRED_ERR  # Red - wrong box or wrong classification
            else:
                color = COLOR_PRED_OK   # Blue - correct
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Build prediction label with error markers
            noun_part = f"{pred_noun_name}"
            if is_noun_wrong:
                noun_part += " ✗"
            verb_part = f"{pred_verb_name}"
            if is_verb_wrong:
                verb_part += " ✗"
            
            pred_label = f"PRED: {noun_part}, {verb_part}, ttc={pred_ttc_str}"
            
            # Draw label below box with background
            label_y = min(H - 20, y2 + 5)
            text_bbox = draw.textbbox((x1, label_y), pred_label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((x1, label_y), pred_label, fill=color)
        
        # Draw rank, IoU, and summary in top-left corner
        summary_lines = [f"#{rank}  IoU={iou:.2f}"]
        if gt_box is None:
            summary_lines.append("⚠ NO GT BOX (all candidates negative)")
        elif is_same_box:
            summary_lines.append("SAME BOX (classification error)")
        else:
            summary_lines.append("DIFFERENT BOX (localization error)")
        if is_noun_wrong:
            summary_lines.append(f"Noun: {pred_noun_name} → {gt_noun_name}")
        if is_verb_wrong:
            summary_lines.append(f"Verb: {pred_verb_name} → {gt_verb_name}")
        if ttc_err > 0.1:
            summary_lines.append(f"TTC err: {ttc_err:.2f}s")
        
        y_pos = 10
        for line in summary_lines:
            text_bbox = draw.textbbox((10, y_pos), line)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((10, y_pos), line, fill=(255, 255, 255))
            y_pos += 20
        
        # Draw legend in top-right corner
        legend_items = [
            ("■ GT (green)", COLOR_GT),
            ("■ Pred OK (blue)", COLOR_PRED_OK),
            ("■ Same box ERR (yellow)", COLOR_SAME_BOX),
            ("■ Diff box ERR (red)", COLOR_PRED_ERR),
        ]
        legend_x = W - 160
        legend_y = 10
        for label, color in legend_items:
            text_bbox = draw.textbbox((legend_x, legend_y), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                          fill=(0, 0, 0, 200))
            draw.text((legend_x, legend_y), label, fill=color)
            legend_y += 18
        
        # Save
        img.save(str(output_path), quality=95)
        return output_path
        
    except Exception as e:
        print(f"[error_analysis] Error creating overlay: {e}")
        return None


def _create_failure_summary(
    failures: List[Dict],
    gallery_dir: Path,
    noun_names: Dict[int, str],
    verb_names: Dict[int, str],
) -> None:
    """Create a text summary of all failure cases."""
    summary_path = gallery_dir / "failure_summary.txt"
    
    lines = [
        "=" * 70,
        "FAILURE CASE GALLERY - ERROR SUMMARY",
        "=" * 70,
        "",
        "Legend:",
        "  - Green box: Ground Truth (GT)",
        "  - Blue box: Correct prediction", 
        "  - Red box: Wrong prediction (noun or verb error)",
        "",
        "Files are named: failure_01.jpg, failure_02.jpg, etc.",
        "Ranked by 'badness' score (noun errors weighted 2x, verb errors 1x)",
        "",
        "-" * 70,
        "",
    ]
    
    for i, f in enumerate(failures):
        lines.append(f"#{i+1:02d} - badness={f.get('_badness', 0):.2f}")
        lines.append(f"    UID: {f.get('uid', '?')}")
        lines.append(f"    Frame: {f.get('frame_path', '?')}")
        
        gt_noun = f.get("_gt_noun_name", "?")
        pred_noun = f.get("_pred_noun_name", "?")
        noun_status = "✗ WRONG" if f.get("_is_noun_wrong") else "✓"
        lines.append(f"    Noun: pred={pred_noun}, gt={gt_noun} {noun_status}")
        
        gt_verb = f.get("_gt_verb_name", "?")
        pred_verb = f.get("_pred_verb_name", "?")
        verb_status = "✗ WRONG" if f.get("_is_verb_wrong") else "✓"
        lines.append(f"    Verb: pred={pred_verb}, gt={gt_verb} {verb_status}")
        
        try:
            gt_ttc = float(f.get("gt_ttc", f.get("ttc_gt", 0)))
            pred_ttc = float(f.get("pred_ttc", f.get("ttc_pred", 0)))
            lines.append(f"    TTC: pred={pred_ttc:.2f}s, gt={gt_ttc:.2f}s, err={abs(pred_ttc-gt_ttc):.2f}s")
        except:
            lines.append(f"    TTC: pred={f.get('pred_ttc', '?')}, gt={f.get('gt_ttc', '?')}")
        
        lines.append("")
    
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    
    print(f"[error_analysis] Saved failure summary: {summary_path}")


# ===================== Plotting Functions =====================

def plot_per_class_accuracy(
    analysis: Dict[str, Any],
    output_path: Path,
    show_worst: bool = True,
) -> None:
    """Plot per-class accuracy bar chart."""
    if not HAS_MATPLOTLIB:
        return
    
    class_type = analysis["class_type"]
    classes = analysis["worst_classes"] if show_worst else analysis["best_classes"]
    title = f"{'Worst' if show_worst else 'Best'} {class_type.title()} Classes by Accuracy"
    
    if not classes:
        print(f"[error_analysis] No classes to plot for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [c["name"][:20] for c in classes]  # Truncate long names
    accuracies = [c["accuracy"] * 100 for c in classes]
    totals = [c["total"] for c in classes]
    
    # Color by accuracy
    colors = plt.cm.RdYlGn([a / 100 for a in accuracies])
    
    bars = ax.barh(range(len(names)), accuracies, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add sample counts
    for i, (bar, total) in enumerate(zip(bars, totals)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"n={total}", va="center", fontsize=8)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 110)  # Leave room for labels
    ax.axvline(x=analysis["overall_accuracy"] * 100, color="red", linestyle="--", 
               label=f"Overall: {analysis['overall_accuracy']*100:.1f}%")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[error_analysis] Saved: {output_path}")
    
    # Save plot data
    save_plot_data({
        "plot_type": "per_class_accuracy",
        "show_worst": show_worst,
        "classes": [{"name": n, "accuracy": float(a), "total": int(t)} 
                    for n, a, t in zip(names, accuracies, totals)],
        "overall_accuracy": float(analysis["overall_accuracy"])
    }, output_path.stem, output_path.parent)


def plot_box_size_analysis(
    analysis: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot accuracy by box size."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ["small_objects", "medium_objects", "large_objects"]
    labels = ["Small\n(<1% area)", "Medium\n(1-5% area)", "Large\n(>5% area)"]
    accuracies = [analysis[cat]["accuracy"] * 100 for cat in categories]
    totals = [analysis[cat]["total"] for cat in categories]
    
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77"]
    
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=1)
    
    # Add sample counts
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={total}", ha="center", fontsize=10)
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Object Size")
    ax.set_ylim(0, max(accuracies) * 1.2 if accuracies else 100)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[error_analysis] Saved: {output_path}")
    
    # Save plot data
    save_plot_data({
        "plot_type": "box_size_accuracy",
        "categories": [{"label": l, "accuracy": float(a), "total": int(t)} 
                      for l, a, t in zip(labels, accuracies, totals)]
    }, output_path.stem, output_path.parent)


def plot_ttc_error_distribution(
    analysis: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot TTC error histogram."""
    if not HAS_MATPLOTLIB:
        return
    
    errors = analysis.get("errors", [])
    if not errors:
        print("[error_analysis] No TTC errors to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(errors, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(x=analysis["mean_error"], color="red", linestyle="--",
                label=f"Mean: {analysis['mean_error']:.3f}s")
    ax1.axvline(x=analysis["median_error"], color="orange", linestyle="--",
                label=f"Median: {analysis['median_error']:.3f}s")
    ax1.set_xlabel("TTC Error (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("TTC Error Distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Error bins bar chart
    bins = analysis.get("error_bins", {})
    if bins:
        ax2.bar(bins.keys(), bins.values(), color="coral", edgecolor="black")
        ax2.set_xlabel("Error Range")
        ax2.set_ylabel("Count")
        ax2.set_title("TTC Errors by Range")
        ax2.grid(axis="y", alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[error_analysis] Saved: {output_path}")
    
    # Save plot data
    save_plot_data({
        "plot_type": "ttc_error_distribution",
        "errors": [float(e) for e in errors],
        "mean_error": float(analysis["mean_error"]),
        "median_error": float(analysis["median_error"]),
        "error_bins": {k: int(v) for k, v in bins.items()}
    }, output_path.stem, output_path.parent)


def plot_confusion_summary(
    analysis: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot top confusion pairs."""
    if not HAS_MATPLOTLIB:
        return
    
    top_confusions = analysis.get("top_confusions", [])
    if not top_confusions:
        print("[error_analysis] No confusions to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"{c['gt_name'][:15]} → {c['pred_name'][:15]}" for c in top_confusions]
    counts = [c["count"] for c in top_confusions]
    
    bars = ax.barh(range(len(labels)), counts, color="indianred", edgecolor="black", linewidth=0.5)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Confusion Count")
    ax.set_title("Top Noun Confusion Pairs (GT → Predicted)")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[error_analysis] Saved: {output_path}")
    
    # Save plot data
    save_plot_data({
        "plot_type": "confusion_matrix",
        "top_confusions": [{"gt_name": c["gt_name"], "pred_name": c["pred_name"], "count": int(c["count"])} 
                          for c in top_confusions]
    }, output_path.stem, output_path.parent)



# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="Track B Error Analysis")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to metrics JSON file (default: latest)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to predictions CSV file (default: latest)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for analysis results")
    parser.add_argument("--top_k", type=int, default=15,
                        help="Number of top/bottom classes to show")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="Minimum samples for a class to be included")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--no_gallery", action="store_true",
                        help="Skip copying failure overlays")
    args = parser.parse_args()
    
    cfg = AnalysisConfig()
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    cfg.top_k = args.top_k
    cfg.min_samples = args.min_samples
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        if args.metrics:
            metrics_path = Path(args.metrics)
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics, metrics_path = load_latest_metrics(cfg.metrics_dir)
    except FileNotFoundError as e:
        print(f"[error_analysis] Error: {e}")
        return
    
    try:
        if args.predictions:
            predictions_path = Path(args.predictions)
            predictions = []
            with predictions_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions.append(row)
        else:
            predictions, predictions_path = load_latest_predictions(cfg.predictions_dir)
    except FileNotFoundError:
        print("[error_analysis] Warning: No predictions file found. Some analyses will be skipped.")
        predictions = []
    
    # Load class names
    noun_names = load_noun_names(cfg.noun_mapping_path)
    verb_names = load_verb_names(cfg.noun_mapping_path)  # Often in same file
    
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS REPORT")
    print(f"{'='*60}\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "metrics_file": str(metrics_path) if 'metrics_path' in dir() else None,
        "predictions_file": str(predictions_path) if predictions else None,
    }
    
    # 1. Per-noun accuracy analysis
    print("1. PER-NOUN ACCURACY ANALYSIS")
    print("-" * 40)
    if "per_noun_stats" in metrics:
        noun_analysis = analyze_per_class_accuracy(
            metrics["per_noun_stats"], noun_names, "noun",
            cfg.top_k, cfg.min_samples
        )
        results["noun_analysis"] = noun_analysis
        
        print(f"   Total noun classes: {noun_analysis['total_classes']}")
        print(f"   Classes with >={cfg.min_samples} samples: {noun_analysis['classes_with_min_samples']}")
        print(f"   Overall noun accuracy: {noun_analysis['overall_accuracy']*100:.2f}%")
        print(f"   Zero-accuracy classes: {len(noun_analysis['zero_accuracy_classes'])}")
        print(f"   Perfect-accuracy classes: {len(noun_analysis['perfect_accuracy_classes'])}")
        
        print("\n   Worst 5 noun classes:")
        for c in noun_analysis["worst_classes"][:5]:
            print(f"      {c['name']}: {c['accuracy']*100:.1f}% (n={c['total']})")
        
        if not args.no_plots and HAS_MATPLOTLIB:
            plot_per_class_accuracy(noun_analysis, cfg.output_dir / "noun_worst_classes.png", show_worst=True)
            plot_per_class_accuracy(noun_analysis, cfg.output_dir / "noun_best_classes.png", show_worst=False)
    else:
        print("   No per-noun stats in metrics file")
    
    # 2. Per-verb accuracy analysis
    print("\n2. PER-VERB ACCURACY ANALYSIS")
    print("-" * 40)
    if "per_verb_stats" in metrics:
        verb_analysis = analyze_per_class_accuracy(
            metrics["per_verb_stats"], verb_names, "verb",
            cfg.top_k, cfg.min_samples
        )
        results["verb_analysis"] = verb_analysis
        
        print(f"   Total verb classes: {verb_analysis['total_classes']}")
        print(f"   Classes with >={cfg.min_samples} samples: {verb_analysis['classes_with_min_samples']}")
        print(f"   Overall verb accuracy: {verb_analysis['overall_accuracy']*100:.2f}%")
        
        print("\n   Worst 5 verb classes:")
        for c in verb_analysis["worst_classes"][:5]:
            print(f"      {c['name']}: {c['accuracy']*100:.1f}% (n={c['total']})")
        
        if not args.no_plots and HAS_MATPLOTLIB:
            plot_per_class_accuracy(verb_analysis, cfg.output_dir / "verb_worst_classes.png", show_worst=True)
            plot_per_class_accuracy(verb_analysis, cfg.output_dir / "verb_best_classes.png", show_worst=False)
    else:
        print("   No per-verb stats in metrics file")
    
    # 3. Box size analysis
    print("\n3. BOX SIZE FAILURE ANALYSIS")
    print("-" * 40)
    if predictions:
        box_analysis = analyze_box_size_failures(predictions, cfg.small_object_threshold)
        results["box_size_analysis"] = box_analysis
        
        print(f"   Small objects (<{cfg.small_object_threshold*100:.0f}% area): "
              f"{box_analysis['small_objects']['accuracy']*100:.1f}% acc (n={box_analysis['small_objects']['total']})")
        print(f"   Medium objects: {box_analysis['medium_objects']['accuracy']*100:.1f}% acc (n={box_analysis['medium_objects']['total']})")
        print(f"   Large objects: {box_analysis['large_objects']['accuracy']*100:.1f}% acc (n={box_analysis['large_objects']['total']})")
        
        if not args.no_plots and HAS_MATPLOTLIB:
            plot_box_size_analysis(box_analysis, cfg.output_dir / "box_size_accuracy.png")
    else:
        print("   No predictions available for box size analysis")
    
    # 4. TTC error analysis
    print("\n4. TTC ERROR ANALYSIS")
    print("-" * 40)
    if predictions:
        ttc_analysis = analyze_ttc_errors(predictions)
        results["ttc_analysis"] = {k: v for k, v in ttc_analysis.items() if k != "errors"}  # Don't save full list
        
        print(f"   Total predictions with TTC: {ttc_analysis['total_with_ttc']}")
        print(f"   Mean TTC error: {ttc_analysis['mean_error']:.4f}s")
        print(f"   Median TTC error: {ttc_analysis['median_error']:.4f}s")
        print(f"   Std TTC error: {ttc_analysis['std_error']:.4f}s")
        
        if ttc_analysis["error_bins"]:
            print("\n   Error distribution:")
            for bin_name, count in ttc_analysis["error_bins"].items():
                print(f"      {bin_name}: {count}")
        
        if not args.no_plots and HAS_MATPLOTLIB:
            plot_ttc_error_distribution(ttc_analysis, cfg.output_dir / "ttc_error_distribution.png")
    else:
        print("   No predictions available for TTC analysis")
    
    # 5. Confusion matrix
    print("\n5. NOUN CONFUSION ANALYSIS")
    print("-" * 40)
    if predictions:
        confusion_analysis = build_confusion_matrix(predictions, noun_names)
        results["confusion_analysis"] = confusion_analysis
        
        print(f"   Total confusions: {confusion_analysis['total_confusions']}")
        print(f"   Unique confusion pairs: {confusion_analysis['unique_confusion_pairs']}")
        
        print("\n   Top 5 confusion pairs:")
        for c in confusion_analysis["top_confusions"][:5]:
            print(f"      {c['gt_name']} → {c['pred_name']}: {c['count']} times")
        
        if not args.no_plots and HAS_MATPLOTLIB:
            plot_confusion_summary(confusion_analysis, cfg.output_dir / "noun_confusions.png")
    else:
        print("   No predictions available for confusion analysis")
    
    # 6. Failure case gallery
    print("\n6. FAILURE CASE GALLERY")
    print("-" * 40)
    if predictions and not args.no_gallery:
        failure_cases = select_failure_cases(
            predictions, cfg.overlays_dir, cfg.output_dir, cfg.failure_gallery_count,
            noun_names=noun_names, verb_names=verb_names
        )
        results["failure_cases_count"] = len(failure_cases)
        
        print(f"   Selected {len(failure_cases)} worst failure cases")
        if failure_cases:
            print("\n   Top 5 failures:")
            for i, f in enumerate(failure_cases[:5]):
                print(f"      {i+1}. uid={f.get('uid', '?')}, frame={f.get('frame', '?')}, "
                      f"badness={f.get('_badness', 0):.2f}")
                if f.get("_is_noun_wrong"):
                    print(f"          Noun: {f.get('_pred_noun_name')} (should be: {f.get('_gt_noun_name')})")
                if f.get("_is_verb_wrong"):
                    print(f"          Verb: {f.get('_pred_verb_name')} (should be: {f.get('_gt_verb_name')})")
    else:
        print("   Skipped (no predictions or --no_gallery)")
    
    # 7. Success case gallery
    print("\n7. SUCCESS CASE GALLERY")
    print("-" * 40)
    if predictions and not args.no_gallery:
        success_cases = select_success_cases(
            predictions, cfg.overlays_dir, cfg.output_dir, cfg.failure_gallery_count,
            noun_names=noun_names, verb_names=verb_names
        )
        results["success_cases_count"] = len(success_cases)
        
        print(f"   Selected {len(success_cases)} best success cases")
        if success_cases:
            print("\n   Top 5 successes:")
            for i, s in enumerate(success_cases[:5]):
                print(f"      {i+1}. uid={s.get('uid', '?')}, frame={s.get('frame', '?')}, "
                      f"goodness={s.get('_goodness', 0):.2f}")
                print(f"          Noun: {s.get('_gt_noun_name')} ✓")
                verb_mark = "✓" if s.get("_is_verb_correct") else "✗"
                print(f"          Verb: {s.get('_gt_verb_name')} {verb_mark}")
                ttc_mark = "✓" if s.get("_is_ttc_correct") else ""
                print(f"          TTC error: {s.get('_ttc_error_s', 0):.3f}s {ttc_mark}")
        else:
            print("   No correct predictions found!")
    else:
        print("   Skipped (no predictions or --no_gallery)")
    
    # Save results JSON
    results_path = cfg.output_dir / f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Analysis complete. Results saved to: {cfg.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
