"""Regression checks for small vessels, noise exclusion and held-out patch support."""
from pathlib import Path
import tempfile,json
import numpy as np,cv2
from train_pair_restoration import _sea_patch_statistics,_crop
from maritime_patch_filter import filter_pairs,PatchPolicy
ROOT=Path(__file__).resolve().parents[1]
def main():
    directory=ROOT/"verification";directory.mkdir(exist_ok=True)
    a=np.full((180,180),3/255.,np.float32);a[75:107,82:96]=180/255.
    point=(88.,90.,88.,90.)
    assert _sea_patch_statistics(_crop(a,*point[:2],128))["is_sea"]
    kept,report,policy=filter_pairs(a,a,[point],directory/"small_ship",128,np.eye(3),jitter=6)
    assert kept==[point] and report["ship_points"]==1
    assert not policy.valid_center(8,8)
    holdout=np.zeros(a.shape,np.uint8);holdout[88:92,86:90]=1
    blocked=PatchPolicy(a,a,np.eye(3),128,6,"maritime",holdout)
    assert not blocked.valid_center(*point[:2])
    # A one-sided bright object and weak speckle cannot become a joint vessel.
    sea=np.full_like(a,3/255.);sea[75:107,82:96]=25/255.
    failed=False
    try:PatchPolicy(sea,a,np.eye(3),128,6)
    except RuntimeError:failed=True
    assert failed
    # Coast-only scene remains supported by the exact shared detector.
    coast=np.full((320,320),3/255.,np.float32);coast[110:145,45:275]=100/255.
    c=(150.,125.,150.,125.)
    kept,report,policy=filter_pairs(coast,coast,[c],directory/"coast_only",128,np.eye(3),jitter=6)
    assert kept==[c] and report["shore_points"]==1
    # Restricting core SAR-SIFT computation to a mask must preserve core outputs.
    from sar_registration.sar_sift_core import SARSIFT,SARSIFTConfig
    from masked_sar_detector import MaskedSARSIFT
    from sar_registration.maritime_targets import point_labels
    rng=np.random.default_rng(42);image=rng.integers(0,255,(64,64),dtype=np.uint8)
    support=np.zeros(image.shape,bool);support[16:48,16:48]=True
    cfg=SARSIFTConfig(layers=2,harris_threshold=.001,max_features=None)
    original,desc=SARSIFT(cfg).detect_and_compute(image)
    eligible=point_labels(original,support)>0
    restricted,masked_desc=MaskedSARSIFT(cfg,support).detect_and_compute(image)
    np.testing.assert_array_equal(original[eligible],restricted)
    np.testing.assert_allclose(desc[eligible],masked_desc,rtol=0,atol=1e-6)
    result=dict(passed=True,checks=["small real vessel survives despite >95 percent dark patch",
        "weak sea noise cannot replace an independently detected source target",
        "coast-only support","strict full-window bounds","holdout support exclusion","unchanged endpoint coordinates","masked detector matches original SAR-SIFT coordinates and descriptors"])
    (directory/"regression.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
