"""Use the registration module's unchanged maritime detector for restoration patches."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from sar_registration.maritime_targets import MaritimeTargets, point_labels
from sar_registration.geometry import project

def gray_u8(image):
    return np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)

class PatchPolicy:
    def __init__(self, sensed, reference, transform, patch_size, jitter,
                 mode="maritime", holdout_reference=None, fixed_source_holdout=None):
        self.mode=mode
        self.transform=np.asarray(transform,dtype=float)
        if self.transform.shape==(2,3):
            self.transform=np.vstack([self.transform,[0,0,1]])
        self.targets=MaritimeTargets(gray_u8(reference),gray_u8(sensed),self.transform,30.)
        self.labels=self.targets.source_labels
        self.reference_labels=self.targets.ref_labels
        shape=sensed.shape; h,w=shape
        holdout=np.zeros(reference.shape,np.uint8) if holdout_reference is None else np.asarray(holdout_reference,np.uint8)
        mapped=cv2.warpAffine(holdout,np.linalg.inv(self.transform)[:2],(w,h),flags=cv2.INTER_NEAREST)
        if fixed_source_holdout is not None: mapped|=np.asarray(fixed_source_holdout,np.uint8)
        self.holdout_source=mapped
        self.radius=int(patch_size//2+jitter+2)
        kernel=np.ones((2*self.radius+1,)*2,np.uint8)
        self.forbidden=cv2.dilate(mapped,kernel)>0
        valid=cv2.warpAffine(np.ones(reference.shape,np.uint8),np.linalg.inv(self.transform)[:2],(w,h),flags=cv2.INTER_NEAREST)
        self.complete=cv2.erode(valid,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0

    def target_id(self, point):
        return int(point_labels(np.asarray(point).reshape(1,2),self.labels)[0])

    def valid_center(self,sx,sy,identity=None):
        x,y=int(round(sx)),int(round(sy))
        h,w=self.labels.shape
        if not(0<=x<w and 0<=y<h and self.complete[y,x] and not self.forbidden[y,x]):
            return False
        if self.mode=="maritime":
            sl=self.target_id([sx,sy])
            mapped=project(np.array([[sx,sy]]),self.transform)
            rl=int(point_labels(mapped,self.reference_labels)[0])
            return sl>0 and sl==rl and (identity is None or sl==identity)
        return True

    def jitter_center(self,sx,sy,dx,dy):
        identity=self.target_id([sx,sy])
        if self.valid_center(sx+dx,sy+dy,identity):
            return sx+dx,sy+dy
        return sx,sy

    def groups(self,points):
        if self.mode!="maritime":return None
        result={}
        for index,(sx,sy,rx,ry) in enumerate(points):
            result.setdefault(self.target_id([sx,sy]),[]).append(index)
        return list(result.values())

def filter_pairs(sensed,reference,points,output_dir,patch_size,transform,
                 mode="maritime",jitter=6,holdout_reference_path=None,
                 fixed_source_holdout_path=None):
    if transform is None:
        raise ValueError("Maritime restoration requires a sensed-to-reference transform")
    holdout=np.load(holdout_reference_path) if holdout_reference_path else None
    fixed=np.load(fixed_source_holdout_path) if fixed_source_holdout_path else None
    policy=PatchPolicy(sensed,reference,transform,patch_size,jitter,mode,holdout,fixed)
    source=np.asarray([[p[0],p[1]] for p in points]).reshape(-1,2)
    target=np.asarray([[p[2],p[3]] for p in points]).reshape(-1,2)
    sl=point_labels(source,policy.labels)
    rl=point_labels(target,policy.reference_labels)
    kept=[]; records=[]
    for i,p in enumerate(points):
        same=bool(sl[i]>0 and sl[i]==rl[i])
        if mode=="maritime" and not same: reason="not_same_independently_detected_target"
        elif not policy.valid_center(p[0],p[1]): reason="incomplete_window_holdout_overlap_or_mapped_center_outside_target"
        else: reason=""
        if not reason:kept.append(p)
        records.append(dict(input_index=i,sensed_x=p[0],sensed_y=p[1],reference_x=p[2],reference_y=p[3],
                            source_target_id=int(sl[i]),reference_target_id=int(rl[i]),accepted=not bool(reason),reason=reason))
    counts={}
    for p in kept:
        identity=policy.target_id(p[:2]);counts[str(identity)]=counts.get(str(identity),0)+1
    report=dict(mode=mode,input_pairs=len(points),kept_pairs=len(kept),removed_pairs=len(points)-len(kept),
                records=records,points_per_target=counts,
                ship_points=sum(c for i,c in counts.items() if 0<int(i)<100),
                shore_points=sum(c for i,c in counts.items() if int(i)>=100),
                shared_detector="sar_registration.maritime_targets.MaritimeTargets",
                coordinates_preserved=True,patch_radius_with_jitter_and_interpolation=policy.radius,
                boundary_policy="full real-image support; exclude any evaluation target pixels",
                target_balanced_sampling=mode=="maritime")
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    policy.targets.save(directory/"maritime_targets")
    (directory/"maritime_patch_filter.json").write_text(json.dumps(report,indent=2))
    return kept,report,policy
