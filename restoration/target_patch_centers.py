"""Real target SAR-SIFT patch centers; mapped supervision coordinates are not descriptor matches."""
from pathlib import Path
import csv,json,hashlib
import cv2,numpy as np
from fusion.sar_sift import SarSiftConfig,run_sar_sift as coarse_match,_draw_matches
from sar_registration.maritime_targets import MaritimeTargets,point_labels
from sar_registration.sar_sift_core import SARSIFT,SARSIFTConfig
from sar_registration.geometry import project,robust_affine
from registration_metrics import calculate_registration_metrics
from masked_sar_detector import MaskedSARSIFT

def roi_features(image,labels,identity,maximum=180):
    ys,xs=np.where(labels==identity);pad=96
    x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad)
    x1=min(labels.shape[1],int(xs.max())+pad+1);y1=min(labels.shape[0],int(ys.max())+pad+1)
    configs=[(1.,1.8,.001,8),(1.,2.4,.01,5),(2.,1.8,.01,3),(.5,2.,.8,8)] if identity<100 else [(1.,1.8,.01,4),(.5,2.,.8,8)]
    points=[];descs=[];provenance=[];occupied=set()
    for scale,sigma,threshold,layers in configs:
        roi=image[y0:y1,x0:x1]
        if scale!=1.:roi=cv2.resize(roi,(round(roi.shape[1]*scale),round(roi.shape[0]*scale)))
        yy,xx=np.mgrid[:roi.shape[0],:roi.shape[1]]
        mapped_grid=np.column_stack((((xx+.5)/scale-.5+x0).ravel(),((yy+.5)/scale-.5+y0).ravel()))
        support=(point_labels(mapped_grid,labels)==identity).reshape(roi.shape)
        p,d=MaskedSARSIFT(SARSIFTConfig(sigma=sigma,layers=layers,harris_threshold=threshold,max_features=1500),support).detect_and_compute(roi)
        mapped=(np.asarray(p,float).reshape(-1,2)+.5)/scale-.5+[x0,y0]
        for j in np.flatnonzero(point_labels(mapped,labels)==identity):
            cell=tuple(np.floor(mapped[j]/1.5).astype(int))
            if cell in occupied:continue
            occupied.add(cell);points.append(mapped[j]);descs.append(d[j])
            provenance.append([identity,*mapped[j],scale,sigma,threshold])

    if len(points)>maximum:
        cells={}
        for i,p in enumerate(points):cells.setdefault(tuple(np.floor(np.asarray(p)/24).astype(int)),[]).append(i)
        selected=[]
        while len(selected)<maximum:
            changed=False
            for group in cells.values():
                if group and len(selected)<maximum:selected.append(group.pop(0));changed=True
            if not changed:break
        points=[points[i] for i in selected];descs=[descs[i] for i in selected];provenance=[provenance[i] for i in selected]
    return np.asarray(points).reshape(-1,2),np.asarray(descs).reshape(-1,136),np.asarray(provenance).reshape(-1,6)

def prepare_patch_centers(sensed_path,reference_path,output_dir,transform,target_source_path=None):
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(sensed_path),0);reference=cv2.imread(str(reference_path),0)
    actual=cv2.imread(str(target_source_path or sensed_path),0)
    initial=np.asarray(transform);initial=np.vstack([initial,[0,0,1]]) if initial.shape==(2,3) else initial
    targets=MaritimeTargets(reference,actual,initial,30.)
    fingerprint=hashlib.sha256(image.tobytes()+reference.tobytes()+actual.tobytes()+initial.tobytes()+Path(__file__).read_bytes()+Path('masked_sar_detector.py').read_bytes()).hexdigest()
    cache=directory/'patch_center_report.json'
    if cache.exists() and (directory/'patch_centers.csv').exists():
        saved=json.loads(cache.read_text())
        if saved.get('input_fingerprint')==fingerprint:return saved
    points=[];provenance=[];counts={}
    for identity in targets.active_ids:
        found,_,origin=roi_features(image,targets.source_labels,identity)
        points.extend(found.tolist());provenance.extend(origin.tolist());counts[str(identity)]=len(found)
    source=np.asarray(points).reshape(-1,2);mapped=project(source,initial)
    source_id=point_labels(source,targets.source_labels);reference_id=point_labels(mapped,targets.ref_labels)
    keep=(source_id>0)&(source_id==reference_id)
    source=source[keep];mapped=mapped[keep];provenance=np.asarray(provenance)[keep];identities=source_id[keep]
    if len(source)==0:raise RuntimeError('No real target feature centers visible in both images; no sea fallback')
    fields=['sensed_x','sensed_y','reference_x','reference_y','target_id','sample_kind']
    with (directory/'patch_centers.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for s,r,i in zip(source,mapped,identities):writer.writerow(dict(sensed_x=float(s[0]),sensed_y=float(s[1]),reference_x=float(r[0]),reference_y=float(r[1]),target_id=int(i),sample_kind='real_source_detector_center_with_affine_resampled_reference_supervision'))
    np.savetxt(directory/'best_T0.txt',initial[:2]);np.savez_compressed(directory/'detector_provenance.npz',sensed=provenance)
    targets.save(directory/'maritime_targets',source,mapped)
    report=dict(input_fingerprint=fingerprint,detected_per_target=counts,candidate_centers=len(source),
        points_per_target={str(i):int((identities==i).sum()) for i in targets.active_ids},
        sample_kind='real source SAR-SIFT points, T-mapped reference sampling centers, NOT new descriptor matches or inliers',
        transform=initial.tolist(),target_source_image=str(target_source_path or sensed_path),coordinates_preserved=True)
    cache.write_text(json.dumps(report,indent=2));return report
