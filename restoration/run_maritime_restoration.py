"""Generate two same-grid restored images with the shared maritime target detector."""
from pathlib import Path
from dataclasses import replace
import argparse,json,time,shutil
import cv2,numpy as np,torch
from fusion.sar_sift import run_sar_sift,SarSiftConfig
from target_patch_centers import prepare_patch_centers
from train_pair_restoration import build_parser as training_parser,train_pair_restoration
from fusion.radiometric_alignment import align_radiometry
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",required=True);p.add_argument("--sensed",required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--iterations",type=int,default=300)
    p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--full-resolution-retry",action="store_true",help="Retry strict SAR-SIFT at full resolution if half-size coarse geometry has fewer than six inliers")
    p.add_argument("--keep-previous-geometry",action="store_true",help="Retain a previously reliable coarse transform if restored images on the same pixel grid have too few new matches")
    a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(a.sensed).resolve();reference=Path(a.reference).resolve()
    current=source;checkpoint=None;history=[];previous_geometry_dir=None;cv2.setNumThreads(4);torch.set_num_threads(4)
    cfg=SarSiftConfig(ratio=.9,ransac_threshold=5.,fsc_threshold=3.,max_features=6000,max_matches=400)
    for stage in range(1,4):
        cv2.setRNGSeed(a.seed)
        rd=out/f"round{stage}_sarsift"
        matching=run_sar_sift(current,reference,rd,cfg,f"round{stage}")
        entry=dict(round=stage,matching=matching)
        if stage<=2:
            if matching['inlier_matches']<6 and a.full_resolution_retry:
                entry['half_resolution_attempt']=matching
                retry_dir=out/f"round{stage}_sarsift_full_resolution"
                print(f"Round {stage}: only {matching['inlier_matches']} coarse inliers; retry full resolution",flush=True)
                retry=run_sar_sift(current,reference,retry_dir,replace(cfg,processing_scale=1.),f"round{stage}_full_resolution")
                entry['full_resolution_attempt']=retry
                if retry['inlier_matches']>=6:
                    matching=retry;rd=retry_dir;entry['matching']=retry
                else:
                    (out/f'round{stage}_coarse_failure.json').write_text(json.dumps(entry,indent=2))
            if matching['inlier_matches']<6:
                if previous_geometry_dir is None or not a.keep_previous_geometry:raise RuntimeError('No reliable coarse geometry')
                reused_dir=out/f'round{stage}_geometry_reused'
                reused_dir.mkdir()
                shutil.copy2(previous_geometry_dir/'best_T0.txt',reused_dir/'best_T0.txt')
                geometry=dict(policy='reuse_last_reliable_SAR_SIFT_transform_on_unchanged_pixel_grid',
                              source_transform=str(previous_geometry_dir/'best_T0.txt'),
                              current_stage_inliers=matching['inlier_matches'],not_new_correspondences=True)
                (reused_dir/'geometry_provenance.json').write_text(json.dumps(geometry,indent=2))
                entry['geometry_reuse']=geometry;rd=reused_dir
                print(f'Round {stage}: retain previous reliable transform on same pixel grid; no new correspondences claimed',flush=True)
            else:previous_geometry_dir=rd
            pd=out/f'round{stage}_target_centers'
            entry['target_patch_centers']=prepare_patch_centers(current,reference,pd,np.loadtxt(rd/'best_T0.txt'),target_source_path=source)
            dd=out/f"restoration_after_round{stage}"
            args=training_parser().parse_args(["--sensed",str(source),"--reference",str(reference),
                "--matched-points",str(pd/"patch_centers.csv"),"--transform-file",str(rd/"best_T0.txt"),
                "--output-dir",str(dd),"--device",a.device,"--iterations",str(a.iterations),
                "--seed",str(a.seed),"--target-filter","maritime"])
            args.save_training_patches=True
            if checkpoint:args.init_checkpoint=checkpoint;args.learning_rate=5e-5;args.teacher_weight=.5
            restoration=train_pair_restoration(args)
            if not restoration["trained"]:raise RuntimeError("No reliable jointly visible target patches; no sea fallback")
            entry["restoration"]=restoration;checkpoint=restoration["checkpoint"]
            labels=np.load(dd/"maritime_targets/target_instances.npz")["reference_labels"]
            corrected=dd/"restored_sensed_radiometric.png"
            entry["radiometry"]=align_radiometry(dd/"restored_sensed.png",reference,rd/"best_T0.txt",corrected,fit_mask=labels>0)
            current=corrected
        history.append(entry)
        (out/"restoration_summary.json").write_text(json.dumps(dict(reference=str(reference),sensed=str(source),
            history=history,source_policy="original_sensed_for_both_rounds",same_pixel_grid=True),indent=2))
    print(json.dumps(dict(restored_1=str(out/"restoration_after_round1/restored_sensed_radiometric.png"),
                          restored_2=str(out/"restoration_after_round2/restored_sensed_radiometric.png")),indent=2))
if __name__=="__main__":main()
