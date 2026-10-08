"""Uncertainty-gated reciprocal descriptors + neighborhood displacement consensus."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sar_registration.geometry import project, robust_affine, coverage
from sar_registration.simclr import describe, valid_centers
from sar_registration.geometry import weighted_affine, sane
from sar_registration.maritime_targets import point_labels
from registration_metrics import predictive_loo_rmse


def select_distributed(points, scores, shape, maximum=24, minimum_spacing=32.):
    """Confidence/diversity selection, never sort by the reported fit error.

    Geometry has already rejected gross outliers. The final budget is selected
    by learned confidence and spatial diversity, with a half-budget quadrant cap.
    """
    if len(points) == 0:
        return np.empty(0,int)
    h,w = shape[:2]
    quadrant = (points[:,0] >= w/2).astype(int)+2*(points[:,1] >= h/2)
    selected, remaining, counts = [],np.ones(len(points),bool),np.zeros(4,int)
    norm = max(float(np.max(scores)),1e-8)
    while len(selected) < min(maximum,len(points)):
        allowed = remaining & (counts[quadrant] < int(np.ceil(maximum/2)))
        if selected:
            distances = np.min(np.linalg.norm(points[:,None]-points[selected][None],axis=2),axis=1)
            allowed &= distances >= minimum_spacing
            diversity = .25+.75*np.minimum(distances/(.3*min(h,w)),1.)
        else:
            diversity = np.ones(len(points))
        if not allowed.any():
            break
        merit = (np.asarray(scores)/norm+.05)*diversity/(1.+.2*counts[quadrant])
        index = int(np.argmax(np.where(allowed,merit,-np.inf)))
        selected.append(index);remaining[index]=False;counts[quadrant[index]]+=1
    return np.asarray(selected,int)


def predictive_loo(source, target, scores):
    """Use the same confidence-weighted leave-one-out prediction as reports."""
    return predictive_loo_rmse(source, target, scores)


def reciprocal_candidates(source_desc, reference_desc, projected, reference_points,
                          radius, min_cosine=.5, min_margin=.015,
                          source_instances=None,reference_instances=None):
    # Bounded by candidate caps (default 3000 x 4000); never allocate NxMxD.
    cosine = source_desc @ reference_desc.T
    distance = np.linalg.norm(projected[:, None, :] - reference_points[None, :, :], axis=2)
    allowed = distance <= radius
    if source_instances is not None:
        allowed &= (source_instances[:,None] == reference_instances[None,:]) & (source_instances[:,None]>0)
    # Geometry is a soft prior, not the descriptor identity. Radius remains a
    # hard uncertainty bound to suppress repetitive structures far from T0.
    score = cosine - .10*(distance/max(radius, 1e-6))**2
    score[~allowed] = -np.inf
    row = np.argmax(score, axis=1)
    col = np.argmax(score, axis=0)
    ids = np.arange(len(source_desc))
    best = score[ids, row]
    if score.shape[1] < 2 or score.shape[0] < 2:
        return np.empty((0, 2), int), np.empty(0)
    second_row = np.partition(score, -2, axis=1)[:, -2]
    second_col = np.partition(score, -2, axis=0)[-2]
    # Require a genuine competing descriptor in both directions; an isolated
    # geometric candidate must not receive an infinite confidence margin.
    valid = (np.isfinite(best) & np.isfinite(second_row) & np.isfinite(second_col[row])
        & (col[row] == ids) & (cosine[ids,row] >= min_cosine))
    row_margin = np.zeros(len(ids)); col_margin = np.zeros(len(ids))
    row_margin[valid] = best[valid]-second_row[valid]
    col_margin[valid] = best[valid]-second_col[row[valid]]
    keep = valid & (row_margin >= min_margin) & (col_margin >= min_margin)
    pairs = np.column_stack((ids[keep], row[keep]))
    confidence = cosine[ids[keep],row[keep]] * np.clip(
        np.minimum(row_margin[keep],col_margin[keep])/.1, .05, 1)
    return pairs, confidence


def neighborhood_filter(projected, targets, radius, neighbors=8):
    if len(projected) < 6:
        return np.ones(len(projected), bool)
    _, indices = cKDTree(projected).query(projected, k=min(neighbors+1,len(projected)))
    displacement = targets-projected
    residuals = np.linalg.norm(displacement[:,None,:]-displacement[indices[:,1:]], axis=2)
    tolerance = max(2., radius*.15)
    # Local majority consensus suppresses isolated descriptor matches while
    # admitting smoothly varying affine corrections across a large image.
    return (residuals <= tolerance).mean(axis=1) >= .5


def match_iteratively(model, reference, source_points, reference_points, initial,
                      anchor_source, anchor_target, directory: Path, iterations=3,
                      radius=40., min_cosine=.5, min_margin=.015,
                      batch_size=128, threshold=2., iteration_callback=None,
                      max_matches=24, min_matches=10, image_evidence=None, targets=None):
    """Train once, then T_k -> Xs_k -> descriptors -> matches -> T_(k+1).

    The callback runs for EVERY attempted round, including rejected or empty
    rounds. Only accepted transforms feed the next round's reference crops.
    """
    directory.mkdir(parents=True, exist_ok=True)
    transform = initial.copy()
    reference_desc = describe(model, reference, reference_points, batch_size)
    history, best_pairs = [], np.empty((0, 2), int)
    best_scores, best_mask = np.empty(0), np.empty(0, bool)
    anchor_source, anchor_target = np.asarray(anchor_source), np.asarray(anchor_target)

    def anchor_loss(h):
        e = np.linalg.norm(project(anchor_source, h)-anchor_target, axis=1)
        return float(np.mean(np.minimum(e, 10.)**2)) if len(e) else float('inf')

    baseline_anchor = anchor_loss(initial)

    for iteration in range(iterations):
        previous_transform = transform.copy()
        guidance = None
        if image_evidence is not None and iteration == 0:
            input_transform,guidance = image_evidence.refine(transform)
        else:
            input_transform = transform.copy()
        projected_all = project(source_points, input_transform)
        source_ids = np.flatnonzero(valid_centers(projected_all, reference.shape))
        projected = projected_all[source_ids]
        pairs, scores = np.empty((0, 2), int), np.empty(0)
        mask = np.empty(0, bool)
        fitted, accepted, movement = None, False, None
        record = {'iteration': iteration+1, 'radius': radius,
                  'valid_Xs_count': len(source_ids), 'candidate_pairs': 0,
                  'previous_transform':previous_transform.tolist(), 'image_guidance':guidance}
        if len(projected) < 4 or len(reference_points) < 4:
            record['status'] = 'insufficient_valid_crops'
        else:
            # This call always crops fresh Xs from reference at T_k(source).
            source_desc = describe(model, reference, projected, batch_size)
            pairs, scores = reciprocal_candidates(source_desc, reference_desc, projected,
                reference_points, radius, min_cosine, min_margin,
                **(dict(source_instances=point_labels(source_points[source_ids],targets.source_labels),
                   reference_instances=point_labels(reference_points,targets.ref_labels)) if targets is not None else {}))
            if len(pairs):
                keep = neighborhood_filter(projected[pairs[:, 0]], reference_points[pairs[:, 1]], radius)
                pairs, scores = pairs[keep], scores[keep]
                pairs[:, 0] = source_ids[pairs[:, 0]]
            if targets is not None and len(pairs):
                target_keep=targets.filter_pairs(source_points[pairs[:,0]],reference_points[pairs[:,1]])
                pairs,scores=pairs[target_keep],scores[target_keep]
            record['candidate_pairs'] = len(pairs)
            if len(pairs) < 6:
                record['status'] = 'insufficient_matches'
            else:
                src, dst = source_points[pairs[:, 0]], reference_points[pairs[:, 1]]
                fitted, mask = robust_affine(src, dst, threshold, scores)
                if fitted is None or mask.sum() < min_matches:
                    record['status'] = 'degenerate_fit'
                    fitted = None
                else:
                    pool = np.flatnonzero(mask)
                    chosen = pool[targets.select(dst[pool],scores[pool],max_matches) if targets is not None else
                                  select_distributed(dst[pool],scores[pool],reference.shape,max_matches)]
                    record['geometric_pool_count'] = int(mask.sum())
                    if len(chosen) < min_matches:
                        record['status'] = 'insufficient_distributed_matches'
                        fitted = None
                    else:
                        # Fit the actual exported subset. Evaluate excluded geometric
                        # supporters as well, so a tiny favorable subset cannot drift.
                        try:
                            selected_fit = weighted_affine(src[chosen],dst[chosen],scores[chosen])
                        except (ValueError,np.linalg.LinAlgError):
                            selected_fit = None
                        if not sane(selected_fit):
                            record['status'] = 'degenerate_selected_fit'
                            fitted = None
                        else:
                            fitted = selected_fit
                        supporters = np.setdiff1d(pool,chosen)
                        supporter_rmse = float(np.sqrt(np.mean(np.sum(
                            (project(src[supporters],selected_fit)-dst[supporters])**2,axis=1)))) if len(supporters) and fitted is not None else None
                        loo = predictive_loo(src[chosen],dst[chosen],scores[chosen])
                        record.update(supporter_count=len(supporters),supporter_rmse=supporter_rmse,
                                      predictive_loo_rmse=loo)
                        mask = np.zeros(len(pairs),bool);mask[chosen]=True
                if fitted is not None:
                    before = np.linalg.norm(project(src, input_transform)-dst, axis=1)
                    after = np.linalg.norm(project(src, fitted)-dst, axis=1)
                    old_anchor, new_anchor = anchor_loss(previous_transform), anchor_loss(fitted)
                    image_allowed,image_before,image_after = True,None,None
                    if image_evidence is not None:
                        image_allowed,image_before,image_after = image_evidence.permits(previous_transform,fitted)
                    # Without image evidence retain the original conservative anchor
                    # guard. With independent image evidence tolerate <=1px extra
                    # anchor RMS: restoration/SAR-SIFT anchors are noisy, not truth.
                    anchor_allowed = (new_anchor <= old_anchor+max(.05,.02*old_anchor)) if image_evidence is None else (
                        np.sqrt(new_anchor) <= np.sqrt(baseline_anchor)+1.)
                    heldout_ok = record.get('supporter_rmse') is None or record['supporter_rmse'] <= threshold*1.25
                    loo_ok = record.get('predictive_loo_rmse',float('inf')) <= threshold*1.5
                    target_quality=targets.quality(src[mask],dst[mask],fitted) if targets is not None else None
                    spatial_allowed=coverage(dst[mask],reference.shape)>=.08 if targets is None else (
                        target_quality['same_target_rate']==1. and target_quality['spatially_valid'])
                    record['target_quality']=target_quality
                    accepted = bool(
                        np.mean(after[mask]**2) < np.mean(before[mask]**2)+1e-10
                        and anchor_allowed and image_allowed and heldout_ok and loo_ok
                        and spatial_allowed)
                    record.update(inliers=int(mask.sum()), anchor_before=old_anchor,
                                  anchor_after=new_anchor, image_before=image_before,image_after=image_after,
                                  anchor_guard=bool(anchor_allowed),image_guard=bool(image_allowed),
                                  supporter_guard=bool(heldout_ok),loo_guard=bool(loo_ok),
                                  point_coverage=coverage(dst[mask],reference.shape),
                                  status='accepted' if accepted else 'rejected_by_quality_guard')
                    if accepted:
                        movement = float(np.median(np.linalg.norm(
                            project(src, fitted)-project(src, input_transform), axis=1)))
                        transform, best_pairs, best_scores, best_mask = fitted, pairs, scores, mask
                        radius = max(8., min(radius, 3*float(np.quantile(after[mask], .9))+4))
        source, target = source_points[pairs[:, 0]], reference_points[pairs[:, 1]]
        evaluation_transform = fitted if fitted is not None else input_transform
        # When no valid fit exists, report no fitted inliers; do not make an
        # empty/degenerate learned stage look like a successful SIFT fallback.
        if fitted is None:
            mask = np.zeros(len(pairs), bool)
        record.update(accepted=accepted, movement=movement,
                      input_transform=input_transform.tolist(),
                      candidate_transform=None if fitted is None else fitted.tolist(),
                      output_transform=transform.tolist(),
                      evaluated_transform='candidate' if fitted is not None else 'input_no_valid_fit')
        event = dict(record=record, source=source, target=target, mask=mask, scores=scores,
                     evaluation_transform=evaluation_transform, input_transform=input_transform,
                     output_transform=transform.copy(), source_ids=source_ids,
                     projected=projected, original_points=source_points[source_ids])
        if iteration_callback is not None:
            record['artifacts'] = iteration_callback(event)
        history.append(record)
        # Persist at each round, so an interruption preserves completed output.
        temporary = directory/'matching_iterations.tmp'
        temporary.write_text(json.dumps(history, indent=2), encoding='utf-8')
        temporary.replace(directory/'matching_iterations.json')
        if not accepted or movement < .05:
            break
    return transform, best_pairs, best_scores, best_mask, history
