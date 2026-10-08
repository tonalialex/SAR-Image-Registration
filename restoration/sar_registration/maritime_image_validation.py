"""Fixed maritime target windows with disjoint fit and validation pixels."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import minimize
from sar_registration.geometry import project,sane

class MaritimeImageEvidence:
    def __init__(self,reference,sensed,initial,targets):
        self.targets=targets;self.shape=reference.shape
        self.reference=cv2.GaussianBlur(np.log1p(reference.astype(np.float32)),(0,0),2.)
        self.sensed=cv2.GaussianBlur(np.log1p(sensed.astype(np.float32)),(0,0),2.)
        self.initial=np.asarray(initial).copy()
        self.windows={};self.fit_ids=[];self.holdout_ids=[]
        h,w=reference.shape
        overlap=cv2.warpAffine(np.ones(sensed.shape,np.uint8),initial[:2],(w,h),flags=cv2.INTER_NEAREST)
        for object_index,identity in enumerate(targets.active_ids):
            yy,xx=np.where(targets.ref_labels==identity)
            if targets.kind[identity]=='ship':
                x0=max(0,int(xx.min())-16);x1=min(w,int(xx.max())+17)
                y0=max(0,int(yy.min())-16);y1=min(h,int(yy.max())+17)
                blocks=[(x0,y0,x1,y1,object_index%2)]
            else:
                blocks=[]
                for row,y in enumerate(range(int(yy.min())//112*112,int(yy.max())+1,112)):
                    for col,x in enumerate(range(int(xx.min())//112*112,int(xx.max())+1,112)):
                        x0=max(0,x+8);y0=max(0,y+8);x1=min(w,x+104);y1=min(h,y+104)
                        if (targets.ref_labels[y0:y1,x0:x1]==identity).sum()>=48:
                            blocks.append((x0,y0,x1,y1,(row+col)%2))
            for number,(x0,y0,x1,y1,split) in enumerate(blocks):
                ys,xs=slice(y0,y1),slice(x0,x1)
                if overlap[ys,xs].mean()<.98:continue
                key=str(identity)+':'+str(number);self.windows[key]=(ys,xs)
                (self.holdout_ids if split else self.fit_ids).append(key)
        if len(self.fit_ids)<2 or len(self.holdout_ids)<2:
            self.windows={};self.fit_ids=[];self.holdout_ids=[]
            for identity in targets.active_ids:
                yy,xx=np.where(targets.ref_labels==identity)
                x0=max(0,int(xx.min())-16);x1=min(w,int(xx.max())+17)
                y0=max(0,int(yy.min())-16);y1=min(h,int(yy.max())+17)
                mx=(x0+x1)//2;my=(y0+y1)//2
                for row,(ya,yb) in enumerate([(y0,my-6),(my+6,y1)]):
                    for col,(xa,xb) in enumerate([(x0,mx-6),(mx+6,x1)]):
                        if (xb-xa)*(yb-ya)<64:continue
                        if (targets.ref_labels[ya:yb,xa:xb]==identity).sum()<8:continue
                        key=str(identity)+':quadrant:'+str(row)+str(col)
                        self.windows[key]=(slice(ya,yb),slice(xa,xb))
                        (self.holdout_ids if (row+col)%2 else self.fit_ids).append(key)
        if not self.fit_ids or not self.holdout_ids:
            raise RuntimeError('Insufficient disjoint target image windows for validation')
        self.fit_points=[];self.fit_reference=[];self.groups=[];offset=0
        for key in self.fit_ids:
            ys,xs=self.windows[key];yy,xx=np.mgrid[ys.start:ys.stop:2,xs.start:xs.stop:2]
            a=self.reference[yy,xx].ravel().astype(np.float64);a-=a.mean();a/=max(np.linalg.norm(a),1e-8)
            self.fit_points.append(np.column_stack((xx.ravel(),yy.ravel())))
            self.fit_reference.append(a);self.groups.append(slice(offset,offset+len(a)));offset+=len(a)
        self.fit_points=np.concatenate(self.fit_points);self.baseline=self.score(initial)

    def score(self,transform):
        h,w=self.shape
        warped=cv2.warpAffine(self.sensed,transform[:2],(w,h))
        values={}
        for ship_id,(ys,xs) in self.windows.items():
            a=self.reference[ys,xs].ravel().astype(float);b=warped[ys,xs].ravel().astype(float)
            a-=a.mean();b-=b.mean()
            values[str(ship_id)]=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-8))
        return dict(fit_ncc=float(np.mean([values[str(i)] for i in self.fit_ids])),
            holdout_ncc=float(np.mean([values[str(i)] for i in self.holdout_ids])),
            all_target_ncc=float(np.mean(list(values.values()))),per_target_window_ncc=values,
            fit_windows=self.fit_ids,holdout_windows=self.holdout_ids,valid=True)

    def _matrix(self,parameters):
        h,w=self.shape;center=np.array([w/2,h/2]);q=parameters
        delta=np.eye(3);delta[:2,:2]+=np.array([[q[0]/w,q[1]/h],[q[3]/w,q[4]/h]])
        delta[:2,2]=np.array([q[2],q[5]])+center-delta[:2,:2]@center
        return delta@self.initial

    def _objective(self,parameters):
        inverse=np.linalg.inv(self._matrix(parameters));points=self.fit_points
        x=inverse[0,0]*points[:,0]+inverse[0,1]*points[:,1]+inverse[0,2]
        y=inverse[1,0]*points[:,0]+inverse[1,1]*points[:,1]+inverse[1,2]
        count=len(x);pad=(-count)%128
        x=np.pad(x,(0,pad),mode='edge').astype(np.float32).reshape(-1,128)
        y=np.pad(y,(0,pad),mode='edge').astype(np.float32).reshape(-1,128)
        samples=cv2.remap(self.sensed,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).ravel()[:count]
        corr=[]
        for a,group in zip(self.fit_reference,self.groups):
            b=samples[group].astype(float);b-=b.mean();corr.append(float(a@b/max(np.linalg.norm(b),1e-8)))
        return -float(np.mean(corr))

    def refine(self,transform):
        # Only fit windows optimize the image prior; disjoint windows validate it.
        result=minimize(self._objective,np.zeros(6),method='Powell',bounds=[(-6.,6.)]*6,
            options={'maxiter':35,'xtol':.02,'ftol':1e-6})
        candidate=self._matrix(result.x);before=self.score(transform);after=self.score(candidate)
        source=self.targets.source_centers
        movement=float(np.max(np.linalg.norm(project(source,candidate)-project(source,self.initial),axis=1)))
        accepted=bool(sane(candidate) and movement<10. and after['fit_ncc']>before['fit_ncc']
            and after['holdout_ncc']>=before['holdout_ncc']-.001)
        return candidate if accepted else transform.copy(),dict(status='accepted' if accepted else 'rejected_on_heldout_ships',
            before=before,after=after,max_ship_movement=movement,fit_evaluations=int(result.nfev))

    def permits(self,previous,candidate):
        before=self.score(previous);after=self.score(candidate)
        allowed=(after['holdout_ncc']>=before['holdout_ncc']-.002 and after['all_target_ncc']>=before['all_target_ncc']-.002)
        return bool(allowed),before,after

