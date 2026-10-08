"""Audit saved restoration artifacts, detector provenance and complete held-out support."""
from pathlib import Path
import csv,json,hashlib,ast
import cv2,numpy as np,torch
from train_pair_restoration import NAFNet,_gray
from maritime_patch_filter import PatchPolicy
from sar_registration.geometry import project
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cv2.setNumThreads(4);torch.set_num_threads(4)
    checks={};input_hashes={}
    original=Path('/root/autodl-tmp/sar_registration')
    for name in ['reference.jpg','sensed.jpg','restored_1.png','restored_2.png']:
        p=ROOT/'inputs'/name;assert sha(p)==sha(original/'data/ship'/name)
        input_hashes[name]=sha(p)
    for name in ['maritime_targets.py','sar_sift_core.py']:
        assert sha(ROOT/'code/sar_registration'/name)==sha(original/'sar_registration'/name)
    assert sha(original/'best_experiments/maritime_target_v6_20261002/experiment_bundle.tar.gz')=='5a6fb5f782de4edfa418da419e9f646706001ac79bf9d5f4cd97010a3dd7ced1'
    checks['canonical_target_detector_and_sar_core_unchanged']=True
    checks['original_success_archive_unchanged']=True
    reference=_gray(ROOT/'inputs/reference.jpg');sensed=_gray(ROOT/'inputs/sensed.jpg')
    holdout=np.load(ROOT/'evaluation/holdout_reference.npy');fixed=np.load(ROOT/'evaluation/holdout_source.npy')
    stages={}
    for mode in ['legacy','maritime']:
        summary=json.loads((ROOT/f'results/{mode}/summary.json').read_text())
        assert len(summary)==3 and summary[-1]['round']==3
        for stage in [1,2]:
            folder=ROOT/f'results/{mode}/restoration_after_round{stage}'
            report=json.loads((folder/'restoration_report.json').read_text())
            assert report['trained'] and report['iterations']==300
            losses=json.loads((folder/'loss_history.json').read_text())
            assert len(losses)==300
            assert np.isfinite(losses).all()
            manifest=json.loads((folder/'training_patch_pairs/training_patch_pairs.json').read_text())
            assert manifest['count']==report['matched_points']
            transform=np.loadtxt(report['transform_file'])
            policy=PatchPolicy(sensed,reference,transform,128,6,mode,holdout,fixed)
            inverse=np.linalg.inv(np.vstack([transform,[0,0,1]]))
            sampled=cv2.warpAffine(holdout.astype(np.float32),inverse[:2],(1350,1350),flags=cv2.INTER_LINEAR)>0
            exclude=sampled|fixed.astype(bool)
            ii=cv2.integral(exclude.astype(np.uint8))
            valid=cv2.warpAffine(np.ones(reference.shape,np.float32),inverse[:2],(1350,1350),flags=cv2.INTER_LINEAR)>=1
            bad=cv2.integral((~valid).astype(np.uint8))
            def window_sum(integral,x,y):
                left=x-64;right=x+64;top=y-64;bottom=y+64
                assert 0<=left<right<=1350 and 0<=top<bottom<=1350
                return integral[bottom,right]-integral[top,right]-integral[bottom,left]+integral[top,left]
            original_rows=list(csv.DictReader(Path(report['source_matched_points']).open(encoding='utf-8-sig')))
            coordinates={(float(r['sensed_x']),float(r['sensed_y']),float(r['reference_x']),float(r['reference_y'])) for r in original_rows}
            count=0
            if mode=='maritime':
                provenance=np.load(Path(report['source_matched_points']).parent/'detector_provenance.npz')['sensed']
                actual={(float(r[1]),float(r[2])) for r in provenance}
            for pair in manifest['pairs']:
                p=(pair['sensed_x'],pair['sensed_y'],pair['reference_x'],pair['reference_y'])
                assert p in coordinates
                if mode=='maritime':
                    assert p[:2] in actual
                    np.testing.assert_allclose(project(np.array([p[:2]]),np.vstack([transform,[0,0,1]]))[0],p[2:],atol=1e-8)
                for dx in range(-6,7):
                    for dy in range(-6,7):
                        x,y=policy.jitter_center(*p[:2],dx,dy);x,y=int(round(x)),int(round(y))
                        assert window_sum(ii,x,y)==0
                        assert window_sum(bad,x,y)==0
                        count+=1
                for kind in ['input_patch','target_patch']:
                    patch=cv2.imread(pair[kind],0);assert patch is not None and patch.shape==(128,128)
            state=torch.load(folder/'pair_specific_nafnet.pth',map_location='cpu',weights_only=True)
            model=NAFNet(img_channel=1,width=state['args']['width'],middle_blk_num=1,enc_blk_nums=[1,1,1,4],dec_blk_nums=[1,1,1,1]).eval()
            model.load_state_dict(state['state_dict'],strict=True)
            with torch.inference_mode():
                out=model(torch.from_numpy(sensed[128:256,128:256].copy())[None,None])
                assert tuple(out.shape)==(1,1,128,128) and torch.isfinite(out).all()
            for file in ['restored_sensed.png','restored_sensed_radiometric.png']:
                img=cv2.imread(str(folder/file),0);assert img is not None and img.shape==(1350,1350)
            stages[f'{mode}_{stage}']=dict(patches=manifest['count'],all_jitter_windows_checked=count,
                full_source_and_reference_support=True,heldout_pixel_overlap=0,model_reload_finite=True)
    for p in (ROOT/'code').glob('*.py'):ast.parse(p.read_text())
    result=dict(passed=True,checks=checks,input_sha256=input_hashes,stages=stages,
        patch_center_note='Mapped reference centers are affine sampling coordinates, not independently detected correspondences.')
    (ROOT/'verification/saved_artifact_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
