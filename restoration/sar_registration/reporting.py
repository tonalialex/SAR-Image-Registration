"""Unified reports: exactly the same four metrics for SIFT and final inliers."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from registration_metrics import calculate_registration_metrics
from sar_registration.geometry import project
from sar_registration.sift_fusion import _draw_pair_visualization


def export_final(directory, reference, sensed, transform, source, target, mask, status, scores=None):
    directory.mkdir(parents=True, exist_ok=True)
    errors = np.linalg.norm(project(source,transform)-target,axis=1)
    mask = np.asarray(mask,bool)
    scores = np.ones(len(source)) if scores is None else np.asarray(scores, float)
    metrics = calculate_registration_metrics(target[mask],errors[mask],reference.shape[:2],
                                             source_points=source[mask],weights=scores[mask])
    metrics.update(status=status, tentative_pairs=len(source), retained_pairs=int(mask.sum()),
                   metric_population='retained_geometric_inliers',
                   metric_note='Internal correspondence residuals, not ground-truth registration accuracy',
                   final_transform=transform.tolist())
    # Unselected supporters are not necessarily outliers. Draw the retained
    # correspondences only; preserve every candidate and its flag in the CSV.
    visual = _draw_pair_visualization(reference,sensed,target[mask],source[mask],
                                     np.ones(int(mask.sum()),bool),
                                     f'Final: {status}, retained={int(mask.sum())}, candidates={len(source)}')
    Image.fromarray(visual).save(directory/'final_registration_matches.png')
    registered = cv2.warpPerspective(sensed,transform,(reference.shape[1],reference.shape[0]))
    Image.fromarray(registered).save(directory/'registered_sensed.png')
    Image.fromarray(cv2.addWeighted(reference,.5,registered,.5,0)).save(directory/'registration_overlay.png')
    np.savetxt(directory/'final_transform.txt',transform,fmt='%.12f')
    np.savetxt(directory/'matched_points.csv',np.column_stack((source,target,mask,errors,scores)),
               delimiter=',',header='sensed_x,sensed_y,reference_x,reference_y,retained,error,confidence',comments='')
    (directory/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'registration_metrics.txt').write_text('\n'.join(f'{k}={metrics[k]}' for k in
        ('Nred','RMSEall','RMSEloo','Pquad'))+'\n',encoding='utf-8')
    return metrics


def comparison(run_dir, output_dir, final_metrics):
    report = json.loads((run_dir/'fusion_report.json').read_text(encoding='utf-8'))
    rows = []
    for label in report['representation_labels']:
        metric = report['sift_metrics'][label]
        rows.append(dict(stage=label, status=metric.get('status','sift_success'),
                         **{k:metric[k] for k in ('Nred','RMSEall','RMSEloo','Pquad')}))
    rows.append(dict(stage='SimCLR_final',status=final_metrics['status'],
                     **{k:final_metrics[k] for k in ('Nred','RMSEall','RMSEloo','Pquad')}))
    with (output_dir/'four_stage_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    (output_dir/'four_stage_metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows

class IterationRecorder:
    """One artifact directory per attempted round; preserve prior matching runs."""
    def __init__(self, output_dir, reference, sensed):
        from datetime import datetime, timezone
        self.directory = output_dir/'iterations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.reference, self.sensed, self.rows = reference, sensed, []
        (output_dir/'latest_iterations.json').write_text(
            json.dumps({'directory':str(self.directory)},indent=2), encoding='utf-8')

    def __call__(self, event):
        record = event['record']
        directory = self.directory/f"iteration_{record['iteration']:03d}"
        metrics = export_final(directory, self.reference, self.sensed,
            event['evaluation_transform'], event['source'], event['target'], event['mask'], record['status'],
            scores=event['scores'])
        # Avoid suggesting the tentative proposal is the accepted final result.
        (directory/'final_registration_matches.png').rename(directory/'registration_matches.png')
        (directory/'final_transform.txt').rename(directory/'evaluated_transform.txt')
        np.savetxt(directory/'input_transform.txt',event['input_transform'],fmt='%.12f')
        np.savetxt(directory/'output_transform.txt',event['output_transform'],fmt='%.12f')
        if record['candidate_transform'] is not None:
            np.savetxt(directory/'candidate_transform.txt',record['candidate_transform'],fmt='%.12f')
        np.savetxt(directory/'Xs_coordinates.csv',np.column_stack((event['source_ids'],
            event['original_points'],event['projected'])), delimiter=',',
            header='sensed_index,sensed_x,sensed_y,reference_crop_x,reference_crop_y',comments='')
        (directory/'iteration.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        metrics.update(accepted=record['accepted'], evaluated_transform=record['evaluated_transform'],
                       input_transform=record['input_transform'],output_transform=record['output_transform'])
        metrics['evaluated_transform_matrix'] = metrics.pop('final_transform')
        (directory/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        self.rows.append(dict(iteration=record['iteration'], status=record['status'],
            accepted=record['accepted'], **{k:metrics[k] for k in ('Nred','RMSEall','RMSEloo','Pquad')}))
        with (self.directory/'iteration_metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(self.rows[0]))
            writer.writeheader();writer.writerows(self.rows)
        (self.directory/'iteration_metrics.json').write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
        return {'directory':str(directory), 'metrics':{k:metrics[k] for k in ('Nred','RMSEall','RMSEloo','Pquad')}}
