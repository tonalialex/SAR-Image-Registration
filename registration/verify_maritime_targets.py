"""Regression checks for maritime masks and genuine feature coordinates."""
import json,cv2,numpy as np
from sar_registration.maritime_targets import MaritimeTargets,stable_brightness,point_labels
from sar_registration.maritime_image_validation import MaritimeImageEvidence
from sar_registration.learned_matching import reciprocal_candidates
def main():
    rng=np.random.default_rng(27);image=rng.integers(0,15,(512,512),dtype=np.uint8)
    for x,y in [(70,60),(165,65),(75,245),(175,270)]:
        image[y:y+75,x:x+15]=rng.integers(170,241,(75,15),dtype=np.uint8)
    image[80:460,375:490]=rng.integers(80,221,(380,115),dtype=np.uint8)
    for x,y in [(250,100),(265,120),(275,150),(260,250)]:image[y,x]=255
    initial=np.eye(3);initial[:2,2]=[12,-8]
    source=cv2.warpAffine(image,np.array([[1,0,-12],[0,1,8]],float),(512,512))
    targets=MaritimeTargets(image,source,initial)
    assert len(targets.ship_ids)>=3 and len(targets.shore_ids)>=1
    noise=np.array([[250,100],[265,120],[275,150],[260,250]],float)
    assert not point_labels(noise,targets.ref_labels).any()
    bright,floor,coast=stable_brightness(image,30.)
    assert not np.any((targets.ref_labels>0)&~bright)
    ship_points=[];shore_points=[]
    for identity in targets.ship_ids:
        y,x=np.where(targets.ref_labels==identity);ship_points.extend(np.column_stack((x,y))[::max(1,len(x)//5)][:5])
    for identity in targets.shore_ids:
        y,x=np.where(targets.ref_labels==identity);shore_points.extend(np.column_stack((x,y))[::max(1,len(x)//40)][:40])
    points=np.array(ship_points+shore_points,float);snapshot=points.copy()
    chosen=targets.select(points,np.linspace(.1,1,len(points)),24)
    labels=point_labels(points[chosen],targets.ref_labels)
    assert np.isin(labels,targets.ship_ids).sum()>=6 and np.isin(labels,targets.shore_ids).sum()>=6
    assert np.array_equal(points,snapshot)
    source_points=points-np.array([12,-8])
    assert targets.filter_pairs(source_points,points).all()
    assert not targets.filter_pairs(source_points,noise[np.zeros(len(points),int)]).any()
    desc=np.eye(len(points));ids=point_labels(points,targets.ref_labels)
    pairs,_=reciprocal_candidates(desc,desc,points,points,200,.5,.005,ids,ids)
    assert len(pairs)==len(points)
    wrong=np.roll(desc,len(ship_points),axis=0)
    pairs,_=reciprocal_candidates(wrong,desc,points,points,1000,.5,.005,ids,ids)
    assert np.all(ids[pairs[:,0]]==ids[pairs[:,1]])
    evidence=MaritimeImageEvidence(image,source,initial,targets)
    assert set(evidence.fit_ids).isdisjoint(evidence.holdout_ids)
    correct=evidence.score(initial);wrong=evidence.score(np.eye(3))
    assert correct['holdout_ncc']>.99 and correct['holdout_ncc']>wrong['holdout_ncc']
    shore_only=np.zeros((512,512),np.uint8);shore_only[80:460,375:490]=image[80:460,375:490]
    coast=MaritimeTargets(shore_only,shore_only,np.eye(3));assert coast.shore_ids and not coast.ship_ids
    MaritimeImageEvidence(shore_only,shore_only,np.eye(3),coast)
    one_ship=np.zeros((512,512),np.uint8);one_ship[60:135,70:85]=image[60:135,70:85]
    vessel=MaritimeTargets(one_ship,one_ship,np.eye(3));assert vessel.ship_ids and not vessel.shore_ids
    MaritimeImageEvidence(one_ship,one_ship,np.eye(3),vessel)
    try:MaritimeTargets(np.zeros((512,512),np.uint8),np.zeros((512,512),np.uint8),np.eye(3))
    except RuntimeError:pass
    else:raise AssertionError('Blank sea must not fabricate target matches')
    print(json.dumps(dict(passed=True,checks=['ship_and_shore_detection','isolated_speckle_rejection','brightness_gate','coordinate_immutability','balanced_ship_shore_selection','same_target_identity','disjoint_image_windows','known_transform_validation','shore_only_scene','single_ship_scene','empty_scene_failure']),indent=2))
if __name__=='__main__':main()

