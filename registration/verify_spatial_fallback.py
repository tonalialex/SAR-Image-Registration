"""Synthetic regression for spatial thinning fallback and retained geometry gates."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
from sar_registration.geometry import project
from sar_registration.learned_matching import match_iteratively

def main():
    image=np.zeros((256,256),np.uint8)
    source=np.array([[80.,80.],[120.,80.],[160.,80.],[80.,120.],
                     [160.,120.],[80.,160.],[120.,160.],[160.,160.]])
    transform=np.eye(3);transform[:2,2]=[2.,1.]
    reference=project(source,transform)
    class Targets:
        ref_labels=np.ones(image.shape,np.int32)
        source_labels=np.ones(image.shape,np.int32)
        selected=5
        def filter_pairs(self,src,dst):return np.ones(len(src),bool)
        def select(self,points,scores,maximum):return np.arange(self.selected)
        def quality(self,src,dst,h):return dict(same_target_rate=1.,spatially_valid=False)
    targets=Targets()
    seen=[]
    def describe(model,img,centers,batch_size):
        seen.append(np.asarray(centers).copy())
        return np.eye(len(centers),dtype=np.float32)
    with TemporaryDirectory(prefix='sar_spatial_policy_') as tmp, \
         patch('sar_registration.learned_matching.describe',side_effect=describe):
        root=Path(tmp)
        final,pairs,scores,mask,history=match_iteratively(None,image,source,reference,np.eye(3),
            root/'fallback',iterations=2,radius=80,threshold=1.3,min_matches=6,targets=targets)
        first=history[0]
        assert first['geometric_pool_count']==8 and first['spatial_selected_count']==5
        assert first['spatial_fallback_used'] and first['fit_pair_count']==8
        assert first['accepted'] and mask.sum()==8 and np.allclose(final,transform)
        assert np.allclose(seen[2],reference), 'Fallback transform did not feed next Xs'
        targets.selected=6
        _,_,_,_,history=match_iteratively(None,image,source,reference,np.eye(3),root/'enough',
            iterations=1,radius=80,threshold=1.3,min_matches=6,targets=targets)
        assert not history[0]['spatial_fallback_used'] and history[0]['fit_pair_count']==6
        assert not history[0]['accepted'], 'Non-fallback spatial quality guard was lost'
        targets.selected=5
        with patch('sar_registration.learned_matching.predictive_loo',return_value=5.):
            _,_,_,_,history=match_iteratively(None,image,source,reference,np.eye(3),root/'bad_loo',
                iterations=1,radius=80,threshold=1.3,min_matches=6,targets=targets)
        assert history[0]['spatial_fallback_used'] and not history[0]['loo_guard'] and not history[0]['accepted']
        with patch('sar_registration.learned_matching.robust_affine',return_value=(transform,np.arange(8)<5)):
            _,_,_,_,history=match_iteratively(None,image,source,reference,np.eye(3),root/'too_few',
                iterations=1,radius=80,threshold=1.3,min_matches=6,targets=targets)
        assert history[0]['status']=='degenerate_fit' and not history[0]['accepted']
    print(json.dumps(dict(passed=True,checks=['under_six_spatial_selection_uses_original_geometric_inliers',
        'fallback_transform_updates_reference_crops','sufficient_selection_keeps_spatial_rules',
        'predictive_loo_still_rejects_bad_geometry','fewer_than_six_geometric_inliers_still_fail']),indent=2))

if __name__=='__main__':main()
