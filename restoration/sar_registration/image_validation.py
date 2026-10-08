"""Disjoint image evidence for a reference-only descriptor registration loop.

Sensed pixels are used only for whole-image affine guidance/validation, never
as neural feature patches. Alternate tiles guide ECC; the other tiles check it.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import minimize
from sar_registration.geometry import project, sane


def _signal(image):
    image = np.log1p(np.asarray(image, np.float32))
    return cv2.GaussianBlur(image, (0, 0), 2.)


class ImageEvidence:
    def __init__(self, reference, sensed, initial):
        self.reference, self.sensed = _signal(reference), _signal(sensed)
        self.shape = reference.shape[:2]
        h, w = self.shape
        self.probes = np.array([[w*.1,h*.1],[w*.9,h*.1],
                                [w*.1,h*.9],[w*.9,h*.9],[w*.5,h*.5]])
        overlap = cv2.warpAffine(np.ones(sensed.shape[:2], np.uint8),
                                 initial[:2], (w,h), flags=cv2.INTER_NEAREST)
        self.tiles = []
        # Nonadjacent central windows leave a safety gap between ECC fit tiles
        # and held-out pixels. Keep the SAME windows for all candidate scores.
        train_mask = np.zeros((h,w), np.uint8)
        step = max(64, min(h,w)//8)
        margin = max(16, step//6)
        for row,y in enumerate(range(step//2, h-step//2, step)):
            for col,x in enumerate(range(step//2, w-step//2, step)):
                ys = slice(y+margin, min(y+step-margin,h))
                xs = slice(x+margin, min(x+step-margin,w))
                block = self.reference[ys,xs]
                if block.size < 256 or block.std() < .12 or overlap[ys,xs].mean() < .99:
                    continue
                split = (row+col)%2
                self.tiles.append((ys,xs,split))
                if split == 0:
                    train_mask[ys,xs] = 255
        # OpenCV's mask is in the INPUT (sensed) grid, not the template grid.
        self.mask = cv2.warpAffine(train_mask, np.linalg.inv(initial)[:2],
                                   (sensed.shape[1],sensed.shape[0]), flags=cv2.INTER_NEAREST)
        self.initial = np.asarray(initial).copy()
        self.baseline = self.score(initial)

    def _polish(self, transform):
        """Optimize fixed fit tiles directly, avoiding moving ECC-mask bias.

        Parameters are displacements in pixels rather than badly scaled affine
        coefficients. Only fit pixels enter the objective; held-out pixels do
        not. Sparse samples make the local six-parameter optimization cheap.
        """
        coordinates,template,groups = [],[],[]
        offset = 0
        for ys,xs,split in self.tiles:
            if split:
                continue
            yy,xx = np.mgrid[ys.start:ys.stop:3,xs.start:xs.stop:3]
            a = self.reference[yy,xx].ravel().astype(np.float64)
            a -= a.mean();a /= max(float(np.linalg.norm(a)),1e-8)
            coordinates.append(np.column_stack((xx.ravel(),yy.ravel())))
            template.append(a);groups.append(slice(offset,offset+len(a)));offset+=len(a)
        points = np.concatenate(coordinates)
        count = len(points)
        # remap requires each map dimension <32767; never use a 1xN map.
        padded = int(np.ceil(count/256))*256
        points = np.pad(points,((0,padded-count),(0,0)),mode='edge')
        h,w = self.shape
        center = np.array([w/2,h/2])
        def matrix(parameters):
            q = parameters
            delta = np.eye(3)
            delta[:2,:2] += np.array([[q[0]/w,q[1]/h],[q[3]/w,q[4]/h]])
            delta[:2,2] = np.array([q[2],q[5]])+center-delta[:2,:2]@center
            return delta@transform
        def objective(parameters):
            inverse = np.linalg.inv(matrix(parameters))
            x = inverse[0,0]*points[:,0]+inverse[0,1]*points[:,1]+inverse[0,2]
            y = inverse[1,0]*points[:,0]+inverse[1,1]*points[:,1]+inverse[1,2]
            samples = cv2.remap(self.sensed,x.astype(np.float32).reshape(-1,256),
                y.astype(np.float32).reshape(-1,256),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).ravel()[:count]
            correlations = []
            for a,group in zip(template,groups):
                b = samples[group].astype(np.float64);b-=b.mean()
                correlations.append(float(a@b/max(float(np.linalg.norm(b)),1e-8)))
            return -float(np.mean(correlations))
        result = minimize(objective,np.zeros(6),method='Powell',bounds=[(-3.,3.)]*6,
                          options={'maxiter':25,'xtol':.02,'ftol':1e-6})
        candidate = matrix(result.x)
        improved = objective(result.x) < objective(np.zeros(6))-1e-6
        return candidate if improved else transform, {'fit_sample_ncc':-float(result.fun),
            'evaluations':int(result.nfev),'improved':bool(improved),'converged':bool(result.success)}

    def score(self, transform):
        h,w = self.shape
        warped = cv2.warpAffine(self.sensed, transform[:2], (w,h))
        valid = cv2.warpAffine(np.ones(self.sensed.shape, np.uint8), transform[:2],
                               (w,h), flags=cv2.INTER_NEAREST)
        values = [[],[]]
        for ys,xs,split in self.tiles:
            if valid[ys,xs].mean() < .99:
                return {'fit_ncc':None,'holdout_ncc':None,'valid':False}
            a,b = self.reference[ys,xs].ravel(),warped[ys,xs].ravel()
            a,b = a-a.mean(),b-b.mean()
            denominator = np.linalg.norm(a)*np.linalg.norm(b)
            values[split].append(float(a@b/max(float(denominator),1e-12)))
        valid_score = min(map(len,values)) >= 4
        return {'fit_ncc':float(np.mean(values[0])) if values[0] else None,
                'holdout_ncc':float(np.mean(values[1])) if values[1] else None,
                'fit_tiles':len(values[0]),'holdout_tiles':len(values[1]),'valid':valid_score}

    def refine(self, transform):
        if not self.baseline['valid']:
            return transform.copy(), {'status':'insufficient_image_tiles'}
        warp = np.linalg.inv(transform)[:2].astype(np.float32)
        try:
            cc, warp = cv2.findTransformECC(self.reference, self.sensed, warp,
                cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,150,1e-6),
                self.mask,5)
            candidate = np.linalg.inv(np.vstack((warp,[0.,0.,1.])))
        except (cv2.error, np.linalg.LinAlgError) as error:
            return transform.copy(), {'status':'ecc_failed','reason':str(error)}
        candidate,polish = self._polish(candidate)
        movement = np.linalg.norm(project(self.probes,candidate)-project(self.probes,self.initial),axis=1)
        # Must stay inside the fit/held-out tile gap, including image corners.
        if not sane(candidate) or movement.max() > 12.:
            return transform.copy(), {'status':'ecc_outside_local_bound'}
        before,after = self.score(transform),self.score(candidate)
        accepted = bool(after['valid'] and after['fit_ncc'] > before['fit_ncc']
                        and after['holdout_ncc'] > before['holdout_ncc']+.001)
        return (candidate if accepted else transform.copy()), {
            'status':'accepted' if accepted else 'rejected_on_holdout',
            'ecc_fit_correlation':float(cc),'before':before,'after':after,
            'max_movement':float(movement.max()),'fixed_fit_polish':polish}

    def permits(self, previous, candidate):
        before,after = self.score(previous),self.score(candidate)
        if not after['valid'] or not before['valid']:
            return False,before,after
        allowed = (after['holdout_ncc'] >= before['holdout_ncc']-.002
                   and after['fit_ncc'] >= before['fit_ncc']-.002
                   and after['holdout_ncc'] > self.baseline['holdout_ncc']+.001)
        return bool(allowed),before,after
