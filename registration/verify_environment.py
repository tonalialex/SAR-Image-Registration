"""CPU-only environment and contract smoke checks; no real-data experiment."""
from __future__ import annotations
import json
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from sar_registration.geometry import project, robust_affine
from sar_registration.simclr import LearningConfig, SARSimCLR, train, describe, crop_reference
from sar_registration.learned_matching import match_iteratively
from sar_registration.learned_matching import select_distributed, predictive_loo
from sar_registration.image_validation import ImageEvidence
from sar_registration.reporting import IterationRecorder, export_final
from sar_registration.sift_fusion import ThreeLayerSIFTFusion
from run_registration import build_parser


def main(write_report=True):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    np.random.seed(7)
    torch.manual_seed(7)
    checks = []
    assert build_parser().parse_args([]).layer_confidences == '1,1,1'
    assert ThreeLayerSIFTFusion._resolve_layer_confidences(3, None) == [1.,1.,1.]
    checks.append('equal_layer_defaults')
    image = np.random.default_rng(7).integers(0,256,(256,256),dtype=np.uint8)
    rgb = np.repeat(image[:,:,None],3,axis=2)
    points = np.array([[80.,80.],[120.,80.],[160.,80.],[80.,120.],
                       [160.,120.],[80.,160.],[120.,160.],[160.,160.]])
    expected = np.array([[1.,0.,2.],[0.,1.,1.],[0.,0.,1.]])
    targets = project(points, expected)
    fitted, mask = robust_affine(points, targets, 2.)
    assert mask.all() and np.allclose(fitted,expected,atol=1e-5)
    checks.append('robust_affine_and_coordinate_direction')
    assert crop_reference(image, targets[0],128).shape == (64,64)
    try:
        crop_reference(image,[0.,0.],128)
        raise AssertionError('Expected boundary failure')
    except ValueError:
        pass
    checks.append('reference_subpixel_crops_and_boundary_guard')
    grid = np.array([(x,y) for y in range(100,901,100) for x in range(100,901,100)],float)
    chosen = select_distributed(grid,np.ones(len(grid)),(1000,1000),24)
    assert len(chosen) == 24
    from sar_registration.geometry import coverage
    assert coverage(grid[chosen],(1000,1000)) > .3
    assert predictive_loo(grid[chosen],project(grid[chosen],expected),np.ones(24)) < 1e-8
    checks.append('distributed_24_point_budget_and_true_leave_one_out_refit')
    texture = cv2.resize(np.random.default_rng(9).integers(10,230,(32,32),dtype=np.uint8),(512,512))
    moving = cv2.warpAffine(texture,np.linalg.inv(expected)[:2],(512,512))
    evidence = ImageEvidence(texture,moving,np.eye(3))
    aligned,diagnostic = evidence.refine(np.eye(3))
    assert diagnostic['status'] == 'accepted',diagnostic
    assert np.max(np.abs(project(grid[:4],aligned)-project(grid[:4],expected))) < .3
    assert evidence.score(aligned)['holdout_ncc'] > evidence.baseline['holdout_ncc']
    checks.append('ecc_direction_and_disjoint_heldout_image_validation')
    with tempfile.TemporaryDirectory(prefix='sar_cpu_verify_') as temporary:
        root = Path(temporary)
        config = LearningConfig(epochs=1,batch_size=4,workers=0,device='cpu',seed=7)
        # One CPU optimizer step on four synthetic instances verifies actual
        # forward/backward, NT-Xent, optimizer, save and resume integration.
        model = train(image,points[:4],root/'model',config)
        first = describe(model,image,points[:4],batch_size=2)
        assert first.shape == (4,256) and np.isfinite(first).all()
        config.resume = True
        resumed = train(image,points[:4],root/'model',config)
        assert np.allclose(first,describe(resumed,image,points[:4],2))
        checks.append('cpu_simclr_forward_backward_checkpoint_resume_descriptor')
        # Deterministic descriptor stub isolates the feedback contract from
        # untrained descriptor accuracy. Real CNN was checked immediately above.
        crops_seen = []
        def deterministic_describe(model,reference,centers,batch_size):
            crops_seen.append(np.asarray(centers).copy())
            return np.eye(len(centers),dtype=np.float32)
        recorder = IterationRecorder(root/'results',rgb,rgb)
        class ImageDiagnosticsStub:
            # ImageEvidence itself was tested on a real known affine above.
            # Isolate acceptance/export types here, including NumPy sqrt bools.
            def refine(self,transform):
                return transform.copy(),{'status':'contract_stub'}
            def score(self,transform):
                # A lower candidate NCC must not veto sound point geometry.
                value = .9 if np.allclose(transform,np.eye(3)) else .1
                return {'holdout_ncc':value,'all_target_ncc':value}
        with patch('sar_registration.learned_matching.describe',side_effect=deterministic_describe):
            final, pairs, scores, inliers, history = match_iteratively(model,image,points,targets,
                np.eye(3),root/'fusion',iterations=3,radius=80.,
                min_cosine=.5,min_margin=.015,threshold=2.,iteration_callback=recorder,min_matches=6,
                image_evidence=ImageDiagnosticsStub())
        assert history[0]['accepted'] and len(history) >= 2
        assert history[0]['image_guard_removed']
        assert history[0]['image_after']['holdout_ncc'] < history[0]['image_before']['holdout_ncc']
        assert np.allclose(crops_seen[1],points)
        assert np.allclose(crops_seen[2],targets,atol=1e-5), 'Xs was not updated after T'
        assert np.allclose(history[1]['input_transform'],history[0]['output_transform'])
        for record in history:
            folder = Path(record['artifacts']['directory'])
            assert (folder/'registration_matches.png').is_file()
            assert (folder/'Xs_coordinates.csv').is_file()
            metric = json.loads((folder/'metrics.json').read_text())
            assert all(k in metric for k in ['Nred','RMSEall','RMSEloo','Pquad'])
        checks.append('accepted_T_feedback_rebuilds_Xs_and_per_round_artifacts')
        # Empty rounds must also export a plot and nullable metrics.
        with patch('sar_registration.learned_matching.describe',side_effect=deterministic_describe), \
             patch('sar_registration.learned_matching.reciprocal_candidates',return_value=(np.empty((0,2),int),np.empty(0))):
            _,_,_,_,empty_history = match_iteratively(model,image,points,targets,
                np.eye(3),root/'empty',iteration_callback=IterationRecorder(root/'empty_results',rgb,rgb))
        assert empty_history[0]['artifacts']['metrics']['Nred'] == 0
        assert empty_history[0]['artifacts']['metrics']['RMSEall'] is None
        metrics = export_final(root/'final',rgb,rgb,final,points[pairs[:,0]],
                               targets[pairs[:,1]],inliers,'cpu_synthetic_check')
        assert metrics['Nred'] >= 6
        checks.append('empty_round_reporting_and_final_exports')
    report = {'status':'PASS','python':platform.python_version(),'torch':torch.__version__,
        'torch_compiled_cuda':torch.version.cuda,'numpy':np.__version__,'opencv':cv2.__version__,
        'scipy':scipy.__version__,'device_used':'cpu','real_data_training':False,
        'gpu_tested':False,'checks':checks}
    print(json.dumps(report,indent=2))
    if write_report:
        Path('environment_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
