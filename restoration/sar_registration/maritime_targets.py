"""Bright stable maritime target constraints; never move or invent keypoints."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sar_registration.geometry import project,coverage

def propose_ships(image, relaxed=False):
    gray = np.asarray(image,np.uint8)
    smooth = cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.)
    outer,guard = (121,61) if relaxed else (81,31)
    area = outer*outer-guard*guard
    mean = (cv2.boxFilter(smooth,-1,(outer,outer),normalize=False)-cv2.boxFilter(smooth,-1,(guard,guard),normalize=False))/area
    variance = (cv2.boxFilter(smooth*smooth,-1,(outer,outer),normalize=False)-cv2.boxFilter(smooth*smooth,-1,(guard,guard),normalize=False))/area-mean*mean
    sigma = 2. if relaxed else 3.
    contrast = 2.5 if relaxed else 3.
    bright = ((smooth>np.maximum(35. if relaxed else 40.,mean+sigma*np.sqrt(np.maximum(variance,1.)))) & (smooth>contrast*np.maximum(mean,3.)))
    binary = cv2.morphologyEx(bright.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers = cv2.connectedComponentsWithStats(binary,8)
    output = np.zeros(gray.shape,np.int32);objects=[]
    for index in range(1,n):
        x,y,w,h,count=map(int,stats[index]);major=max(w,h)
        if count < 300 or count > 8000 or major < 30 or major > 180:continue
        if not relaxed and major/max(1,min(w,h)) < 1.25:continue
        ys,xs = np.where(labels==index)
        hull = cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
        mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,hull,1)
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
        label=len(objects)+1;output[mask>0]=label
        objects.append(dict(id=label,centroid=centers[index].tolist(),box=[x,y,w,h],bright_area=count,hull=hull[:,0,:].tolist()))
    return output,objects

def point_labels(points,labels):
    points=np.asarray(points,float).reshape(-1,2)
    finite=np.isfinite(points).all(1)
    xy=np.rint(np.where(np.isfinite(points),points,0)).astype(int)
    valid=(finite&(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<labels.shape[1])&(xy[:,1]<labels.shape[0]))
    result=np.zeros(len(points),int)
    result[valid]=labels[xy[valid,1],xy[valid,0]]
    return result

def stable_brightness(image,threshold):
    """Threshold first; component extent suppresses isolated sea sparkle."""
    smooth=cv2.GaussianBlur(np.asarray(image,np.float32),(0,0),1.)
    sea=smooth[smooth<=np.quantile(smooth,.6)]
    median=float(np.median(sea));mad=float(np.median(np.abs(sea-median)))
    floor=max(float(threshold),median+5*1.4826*mad)
    bright=smooth>=floor
    connected=cv2.morphologyEx(bright.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    connected=cv2.morphologyEx(connected,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(connected,8)
    coast=np.zeros(image.shape,bool)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>=6000 and max(w,h)>=200:coast|=labels==i
    return bright,floor,coast

class MaritimeTargets:
    """Independent brightness/component masks for ships and extended shore targets."""
    def __init__(self,reference,sensed,initial,brightness_threshold=30.):
        self.reference,self.sensed=reference,sensed
        self.initial=np.asarray(initial);self.requested_threshold=float(brightness_threshold)
        ref_ship,self.reference_objects=propose_ships(reference)
        src_ship,source_objects=propose_ships(sensed,relaxed=True)
        self.ref_labels=np.zeros(reference.shape,np.int32)
        self.source_labels=np.zeros(sensed.shape,np.int32)
        self.correspondences=[];self.kind={}
        self.reference_bright,self.reference_floor,self.reference_coast=stable_brightness(reference,brightness_threshold)
        self.source_bright,self.source_floor,self.source_coast=stable_brightness(sensed,brightness_threshold)
        if self.reference_objects and source_objects:
            rc=np.array([o['centroid'] for o in self.reference_objects])
            sc=np.array([o['centroid'] for o in source_objects])
            distance=np.linalg.norm(project(sc,initial)[:,None]-rc[None,:],axis=2)
            rows,cols=linear_sum_assignment(distance)
            for row,col in zip(rows,cols):
                if distance[row,col]>35.:continue
                identity=int(col)+1
                ref=(ref_ship==identity)&self.reference_bright
                guide=cv2.dilate((ref_ship==identity).astype(np.uint8),np.ones((13,13),np.uint8))
                guide=cv2.warpPerspective(guide,np.linalg.inv(initial),(sensed.shape[1],sensed.shape[0]),flags=cv2.INTER_NEAREST)
                actual=(src_ship==row+1)&(guide>0)&self.source_bright
                if ref.sum()<30 or actual.sum()<30:continue
                self.ref_labels[ref]=identity;self.source_labels[actual]=identity;self.kind[identity]='ship'
                self.correspondences.append(dict(reference_id=identity,source_id=int(row)+1,kind='ship',
                    sensed_centroid=sc[row].tolist(),reference_centroid=rc[col].tolist(),
                    initial_center_distance=float(distance[row,col])))
        n,regions,stats,centers=cv2.connectedComponentsWithStats(self.reference_coast.astype(np.uint8),8)
        for index in range(1,n):
            identity=100+index
            footprint=regions==index
            ref=footprint&self.reference_bright&(self.ref_labels==0)
            guide=cv2.dilate(footprint.astype(np.uint8),np.ones((25,25),np.uint8))
            guide=cv2.warpPerspective(guide,np.linalg.inv(initial),(sensed.shape[1],sensed.shape[0]),flags=cv2.INTER_NEAREST)
            actual=(guide>0)&self.source_coast&self.source_bright&(self.source_labels==0)
            if ref.sum()<300 or actual.sum()<300:continue
            self.ref_labels[ref]=identity;self.source_labels[actual]=identity;self.kind[identity]='shore'
            yy,xx=np.where(actual);sc=np.array([xx.mean(),yy.mean()])
            self.correspondences.append(dict(reference_id=identity,kind='shore',sensed_centroid=sc.tolist(),
                reference_centroid=centers[index].tolist()))
        self.active_ids=sorted(self.kind)
        if not self.active_ids:raise RuntimeError('No jointly visible bright stable ship/shore regions; no sea-point fallback')
        self.ship_ids=[i for i in self.active_ids if self.kind[i]=='ship']
        self.shore_ids=[i for i in self.active_ids if self.kind[i]=='shore']
        self.reference_centers=np.array([o['reference_centroid'] for o in self.correspondences])
        self.source_centers=np.array([o['sensed_centroid'] for o in self.correspondences])
        yy,xx=np.where(self.ref_labels>0)
        self.target_hull_coverage=coverage(np.column_stack((xx,yy)),reference.shape)

    def filter_pairs(self,source,target):
        a,b=point_labels(source,self.source_labels),point_labels(target,self.ref_labels)
        return (a>0)&(a==b)

    def select(self,points,scores,maximum=24,spacing=6.):
        ids=point_labels(points,self.ref_labels);h,w=self.reference.shape
        quadrants=(points[:,0]>=w/2).astype(int)+2*(points[:,1]>=h/2).astype(int)
        selected=[];counts=np.zeros(4,int);per={i:[] for i in self.active_ids}
        def eligible(identity):
            candidates=np.array([j for j in np.flatnonzero(ids==identity) if j not in selected],int)
            separation=spacing if self.kind[identity]=='ship' else max(16.,spacing)
            if per[identity] and len(candidates):
                distances=np.min(np.linalg.norm(points[candidates,None]-points[per[identity]][None],axis=2),axis=1)
                candidates=candidates[distances>=separation]
            return candidates
        # Seed each visible object by confidence, preferring underrepresented quadrants.
        for identity in self.active_ids:
            available=eligible(identity)
            if len(available):
                merit=scores[available]/(1+counts[quadrants[available]])
                index=int(available[np.argmax(merit)]);selected.append(index);per[identity].append(index);counts[quadrants[index]]+=1
        while len(selected)<maximum:
            pools=[eligible(i) for i in self.active_ids];available=np.concatenate(pools)
            if not len(available):break
            represented=np.unique(quadrants[ids>0]);lowest=int(counts[represented].min())
            allowed=counts[quadrants[available]]<=lowest
            if not allowed.any():
                # Do not inflate dense quadrants when a sparse target area is exhausted.
                allowed=counts[quadrants[available]]<lowest+1
            available=available[allowed]
            if not len(available):break
            distance=np.min(np.linalg.norm(points[available,None]-points[selected][None],axis=2),axis=1)
            merit=scores[available]*(.4+.6*np.minimum(distance/120.,1.))/(1+counts[quadrants[available]])
            index=int(available[np.argmax(merit)]);selected.append(index);per[int(ids[index])].append(index);counts[quadrants[index]]+=1
        return np.asarray(selected[:maximum],int)
    
    def quality(self,source,target,transform):
        a=point_labels(source,self.source_labels);b=point_labels(target,self.ref_labels)
        unique,counts=np.unique(b[b>0],return_counts=True)
        ship_points=int(np.isin(b,self.ship_ids).sum());shore_points=int(np.isin(b,self.shore_ids).sum())
        ships=int(sum(i in self.ship_ids for i in unique))
        relative=float(coverage(target,self.reference.shape)/max(self.target_hull_coverage,1e-8))
        kinds_ok=(ship_points>=min(4,len(self.ship_ids)*2) and shore_points>=4) if self.ship_ids and self.shore_ids else True
        spatial=bool(kinds_ok and relative>=.3 and (ships>=min(2,len(self.ship_ids)) if self.ship_ids else shore_points>=6))
        return dict(source_hit_rate=float(np.mean(a>0)) if len(a) else 0.,
            reference_hit_rate=float(np.mean(b>0)) if len(b) else 0.,
            same_target_rate=float(np.mean((a>0)&(a==b))) if len(a) else 0.,
            ship_points=ship_points,shore_points=shore_points,matched_ships=ships,
            points_per_target={str(i):int(c) for i,c in zip(unique,counts)},
            target_relative_coverage=relative,spatially_valid=spatial,
            reference_brightness_threshold=self.reference_floor,source_brightness_threshold=self.source_floor)
    def save(self,directory,source_points=None,reference_points=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(directory/'reference_target_mask.png'),(self.ref_labels>0).astype(np.uint8)*255)
        cv2.imwrite(str(directory/'sensed_target_mask.png'),(self.source_labels>0).astype(np.uint8)*255)
        np.savez_compressed(directory/'target_instances.npz',reference_labels=self.ref_labels,sensed_labels=self.source_labels)
        (directory/'target_instances.json').write_text(json.dumps(dict(reference_objects=self.reference_objects,
            associations=self.correspondences,active_ids=self.active_ids,kinds=self.kind,brightness_thresholds=[self.reference_floor,self.source_floor],mask_method='brightness threshold + CFAR compact vessels / extended shore components; independent masks + T0 association'),indent=2))
        for name,image,labels,points in [('reference',self.reference,self.ref_labels,reference_points),('sensed',self.sensed,self.source_labels,source_points)]:
            canvas=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            for ship_id in self.active_ids:
                contours,_=cv2.findContours((labels==ship_id).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas,contours,-1,(0,255,255),1)
                if contours:
                    x,y,w,h=cv2.boundingRect(max(contours,key=cv2.contourArea));cv2.putText(canvas,self.kind[ship_id]+' '+str(ship_id),(x,max(15,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            if points is not None:
                for p in points:cv2.circle(canvas,tuple(np.rint(p).astype(int)),2,(0,255,0),1)
            cv2.imwrite(str(directory/(name+'_target_overlay.png')),canvas)

def localized_sar_points(images,labels,maximum=1200):
    """Real SAR-Harris maxima at coarse/fine vessel scales and shoreline scales."""
    from sar_registration.sar_sift_core import SARSIFT,SARSIFTConfig
    points=[];occupied=set();provenance=[]
    ship_configurations=[(1.,1.8,.001,8),(1.,2.4,.01,5),(2.,1.8,.01,3),(.5,2.,.8,8)]
    for ship_id in np.unique(labels):
        if not ship_id:continue
        ys,xs=np.where(labels==ship_id);pad=96
        x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad)
        x1=min(labels.shape[1],int(xs.max())+pad+1);y1=min(labels.shape[0],int(ys.max())+pad+1)
        configurations=ship_configurations if ship_id<100 else [(1.,1.8,.01,4),(.5,2.,.8,8)]
        for layer,image in enumerate(images):
            for scale,sigma,threshold,layers in configurations:
                roi=image[y0:y1,x0:x1]
                if scale != 1.:
                    roi=cv2.resize(roi,(round(roi.shape[1]*scale),round(roi.shape[0]*scale)))
                detector=SARSIFT(SARSIFTConfig(sigma=sigma,layers=layers,harris_threshold=threshold,max_features=6000))
                found,_=detector.detect_and_compute(roi)
                found=(np.asarray(found,float).reshape(-1,2)+.5)/scale-.5+[x0,y0]
                for point in found[point_labels(found,labels)==ship_id]:
                    cell=tuple(np.floor(point/1.5).astype(int))
                    if cell in occupied:continue
                    occupied.add(cell);points.append(point)
                    provenance.append([layer,int(ship_id),float(point[0]),float(point[1]),scale,sigma,threshold])
    if len(points)<=maximum:
        return np.asarray(points,float).reshape(-1,2),np.asarray(provenance,float).reshape(-1,7)
    groups={}
    for i,row in enumerate(provenance):groups.setdefault(int(row[1]),[]).append(i)
    selected=[]
    while len(selected)<min(maximum,len(points)):
        for group in groups.values():
            if group and len(selected)<maximum:selected.append(group.pop(0))
    return np.asarray(points,float).reshape(-1,2)[selected],np.asarray(provenance,float).reshape(-1,7)[selected]
