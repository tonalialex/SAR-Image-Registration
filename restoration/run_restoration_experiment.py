"""Controlled two-round restoration comparison on the user's saved maritime pair."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,traceback,time
import cv2,numpy as np,torch
from fusion.sar_sift import run_sar_sift as global_sift,SarSiftConfig
from fusion.radiometric_alignment import align_radiometry
from target_patch_centers import prepare_patch_centers
from sar_registration.maritime_targets import MaritimeTargets
from train_pair_restoration import build_parser as train_parser,train_pair_restoration
from registration_metrics import calculate_registration_metrics
ROOT=Path(__file__).resolve().parents[1]
def write_json(path,data):
    temporary=Path(path).with_suffix(".tmp");temporary.write_text(json.dumps(data,indent=2,default=str));temporary.replace(path)
def stage_metrics(directory):
    rows=list(csv.DictReader((directory/"sift_inlier_matches.csv").open(encoding="utf-8-sig")))
    metrics=calculate_registration_metrics(np.asarray([[float(r["reference_x"]),float(r["reference_y"])] for r in rows]).reshape(-1,2),
                                           np.asarray([float(r["error"]) for r in rows]),(1350,1350),
                                           source_points=np.asarray([[float(r["sensed_x"]),float(r["sensed_y"])] for r in rows]).reshape(-1,2))
    write_json(directory/"registration_metrics.json",metrics)
    return metrics
def fixed_split():
    reference=cv2.imread(str(ROOT/"inputs/reference.jpg"),0);sensed=cv2.imread(str(ROOT/"inputs/sensed.jpg"),0)
    initial=np.vstack([np.loadtxt(ROOT/"results/shared_round1/best_T0.txt"),[0,0,1]])
    targets=MaritimeTargets(reference,sensed,initial,30.)
    holdout_ids=[targets.ship_ids[-1],targets.shore_ids[0]] if targets.ship_ids and targets.shore_ids else [targets.active_ids[-1]]
    directory=ROOT/"evaluation";directory.mkdir(exist_ok=True)
    holdout=cv2.dilate(np.isin(targets.ref_labels,holdout_ids).astype(np.uint8),np.ones((33,33),np.uint8))
    source=cv2.dilate(np.isin(targets.source_labels,holdout_ids).astype(np.uint8),np.ones((33,33),np.uint8))
    mapped=cv2.warpAffine(holdout,np.linalg.inv(initial)[:2],(sensed.shape[1],sensed.shape[0]),flags=cv2.INTER_NEAREST)
    source|=mapped
    np.save(directory/"holdout_reference.npy",holdout);np.save(directory/"holdout_source.npy",source)
    np.save(directory/"reference_labels.npy",targets.ref_labels);np.save(directory/"sensed_labels.npy",targets.source_labels)
    np.savetxt(directory/"fixed_transform.txt",initial)
    fit_mask=cv2.dilate(((targets.ref_labels>0)&~np.isin(targets.ref_labels,holdout_ids)).astype(np.uint8),np.ones((17,17),np.uint8))>0
    fit_mask&=holdout==0
    np.save(directory/"radiometry_fit_mask.npy",fit_mask)
    write_json(directory/"split.json",dict(holdout_target_ids=holdout_ids,active_ids=targets.active_ids,
        policy="same whole targets held out from NAFNet patches, jitter support and radiometric fitting in both arms",
        limitation="same image pair; common coarse geometry uses detected points, not ground truth; not a multi-scene test"))
    return holdout,fit_mask,initial

def audit_stage(stage_dir):
    report=json.loads((stage_dir/"maritime_patch_filter.json").read_text())
    originals=list(csv.DictReader(Path(json.loads((stage_dir/"restoration_report.json").read_text())["source_matched_points"]).open(encoding="utf-8-sig")))
    original_set={(float(x["sensed_x"]),float(x["sensed_y"]),float(x["reference_x"]),float(x["reference_y"])) for x in originals}
    accepted=[x for x in report["records"] if x["accepted"]]
    assert all((x["sensed_x"],x["sensed_y"],x["reference_x"],x["reference_y"]) in original_set for x in accepted)
    if report["mode"]=="maritime":
        assert all(x["source_target_id"]>0 and x["source_target_id"]==x["reference_target_id"] for x in accepted)
    manifest=json.loads((stage_dir/"training_patch_pairs/training_patch_pairs.json").read_text())
    assert manifest["count"]==len(accepted)
    split=json.loads((ROOT/"evaluation/split.json").read_text())
    assert all(x["source_target_id"] not in split["holdout_target_ids"] for x in accepted)
    return dict(coordinates_preserved=True,same_target_required=report["mode"]=="maritime",
                patch_count=len(accepted),holdout_target_centers_excluded=True)

def run_arm(mode,holdout,fit_mask,initial,steps=300):
    arm=ROOT/"results"/mode;arm.mkdir(exist_ok=True)
    source=ROOT/"inputs/sensed.jpg";reference=ROOT/"inputs/reference.jpg"
    current=source;checkpoint=None;summary=[]
    configuration=SarSiftConfig(ratio=.9,ransac_threshold=5.,fsc_threshold=3.,max_features=6000,max_matches=400)
    for stage in range(1,4):
        write_json(ROOT/"STATUS.json",dict(stage="RUNNING",arm=mode,round=stage,shutdown_after_completion=True,results_remote_only=True))
        directory=arm/f"round{stage}_sarsift"
        cv2.setRNGSeed(2026)
        if stage==1:shutil.copytree(ROOT/"results/shared_round1",directory,dirs_exist_ok=True)
        else:
            full=global_sift(current,reference,directory,configuration,f"round{stage}")
            print("FULL_SAR_SIFT",mode,stage,full["inlier_matches"],flush=True)
        metrics=stage_metrics(directory)
        item=dict(round=stage,full_sar_sift=metrics)
        if stage==3:summary.append(item);break
        training_dir=directory
        if mode=="maritime":
            roi_dir=arm/f"round{stage}_target_matches"
            bootstrap=np.vstack([np.loadtxt(directory/"best_T0.txt"),[0,0,1]])
            target_report=prepare_patch_centers(current,reference,roi_dir,bootstrap,target_source_path=source)
            training_dir=roi_dir;item["target_patch_centers"]=target_report
            print("TARGET_PATCH_CENTERS",stage,target_report["candidate_centers"],target_report["points_per_target"],flush=True)
        output=arm/f"restoration_after_round{stage}"
        argv=["--sensed",str(source),"--reference",str(reference),"--matched-points",str(training_dir/("patch_centers.csv" if mode=="maritime" else "sift_inlier_matches.csv")),
              "--output-dir",str(output),"--transform-file",str(training_dir/"best_T0.txt"),"--device","cuda",
              "--iterations",str(steps),"--target-filter",mode,
              "--holdout-reference-mask",str(ROOT/"evaluation/holdout_reference.npy"),
              "--fixed-source-holdout-mask",str(ROOT/"evaluation/holdout_source.npy")]
        args=train_parser().parse_args(argv);args.save_training_patches=True
        if checkpoint:
            args.init_checkpoint=checkpoint;args.learning_rate=5e-5;args.teacher_weight=.5
        restored=train_pair_restoration(args)
        restored["source_matched_points"]=args.matched_points
        write_json(output/"restoration_report.json",restored)
        if not restored["trained"]:raise RuntimeError(f"{mode} restoration round {stage} had no real training patches")
        item["restoration"]=restored
        item["audit"]=audit_stage(output)
        checkpoint=restored["checkpoint"]
        radiometric=align_radiometry(output/"restored_sensed.png",reference,training_dir/"best_T0.txt",
            output/"restored_sensed_radiometric.png",fit_mask=fit_mask)
        item["radiometric"]=radiometric
        current=output/"restored_sensed_radiometric.png"
        summary.append(item);write_json(arm/"summary.json",summary)
        torch.cuda.empty_cache()
    write_json(arm/"summary.json",summary)
    return summary

def ssim_map(a,b):
    a=a.astype(np.float32)/255.;b=b.astype(np.float32)/255.
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    return ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))

def evaluate():
    directory=ROOT/"evaluation";reference=cv2.imread(str(ROOT/"inputs/reference.jpg"),0)
    sensed=cv2.imread(str(ROOT/"inputs/sensed.jpg"),0)
    labels=np.load(directory/"reference_labels.npy");source_labels=np.load(directory/"sensed_labels.npy")
    split=json.loads((directory/"split.json").read_text());h=np.loadtxt(directory/"fixed_transform.txt")
    base_masks={}
    for i in split["active_ids"]:
        base_masks[str(i)]=cv2.dilate((labels==i).astype(np.uint8),np.ones((33,33),np.uint8))>0
    valid=cv2.warpAffine(np.ones(sensed.shape,np.uint8),h[:2],(reference.shape[1],reference.shape[0]),flags=cv2.INTER_NEAREST)>0
    images={"original":ROOT/"inputs/sensed.jpg","historical_saved_round2":ROOT/"inputs/restored_2.png"}
    for mode in ["legacy","maritime"]:
        for stage in [1,2]:
            for kind,file in [("raw","restored_sensed.png"),("radiometric","restored_sensed_radiometric.png")]:
                images[f"{mode}_round{stage}_{kind}"]=ROOT/f"results/{mode}/restoration_after_round{stage}"/file
    sea=cv2.dilate((source_labels>0).astype(np.uint8),np.ones((65,65),np.uint8))==0
    sea[:80]=False;sea[-80:]=False;sea[:,:80]=False;sea[:,-80:]=False
    result={};warped_images={}
    ref_log=cv2.GaussianBlur(np.log1p(reference.astype(np.float32)),(0,0),2.)
    for name,path in images.items():
        img=cv2.imread(str(path),0)
        if img is None or img.shape!=sensed.shape:raise AssertionError(f"Invalid restoration output {path}")
        warped=cv2.warpAffine(img,h[:2],(reference.shape[1],reference.shape[0]))
        warped_images[name]=warped
        image_log=cv2.GaussianBlur(np.log1p(warped.astype(np.float32)),(0,0),2.)
        ss=ssim_map(reference,warped);per={}
        for identity,mask in base_masks.items():
            mask=mask&valid;a=ref_log[mask].astype(float);b=image_log[mask].astype(float)
            a-=a.mean();b-=b.mean();ncc=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
            mse=float(np.mean((reference[mask].astype(float)-warped[mask].astype(float))**2))
            per[identity]=dict(ncc=ncc,reference_consistency_psnr=float(10*np.log10(255**2/max(mse,1e-12))),
                               reference_consistency_ssim=float(np.mean(ss[mask])),pixels=int(mask.sum()))
        held=[str(i) for i in split["holdout_target_ids"]]
        train=[str(i) for i in split["active_ids"] if str(i) not in held]
        def average(ids,key):return float(np.mean([per[i][key] for i in ids]))
        result[name]=dict(path=str(path),per_target=per,holdout_ncc=average(held,"ncc"),fit_target_ncc=average(train,"ncc"),
                          all_target_ncc=average(list(per),"ncc"),holdout_reference_consistency_psnr=average(held,"reference_consistency_psnr"),
                          holdout_reference_consistency_ssim=average(held,"reference_consistency_ssim"),
                          sea_mean=float(img[sea].mean()),sea_std=float(img[sea].std()),sea_above_30_fraction=float((img[sea]>30).mean()),
                          same_grid=True)
    comparison=dict(evaluation_geometry="same fixed original coarse T for all outputs",split=split,results=result,
                    limitation="Reference consistency on a real SAR pair, not PSNR/SSIM against a clean ground-truth image; historical restorations did not use this holdout split.")
    write_json(directory/"restoration_comparison.json",comparison)
    # Compact diagnostic views are generated on the server only.
    names=["original","legacy_round2_radiometric","maritime_round2_radiometric"]
    mosaics=[]
    for identity in split["holdout_target_ids"]:
        yy,xx=np.where(labels==identity);x0=max(0,int(xx.min())-24);x1=min(reference.shape[1],int(xx.max())+25)
        y0=max(0,int(yy.min())-24);y1=min(reference.shape[0],int(yy.max())+25)
        cells=[]
        for title,img in [("reference",reference)]+[(n,warped_images[n]) for n in names]:
            crop=img[y0:y1,x0:x1];factor=min(220/crop.shape[1],200/crop.shape[0])
            resized=cv2.resize(crop,(max(1,round(crop.shape[1]*factor)),max(1,round(crop.shape[0]*factor))))
            canvas=np.zeros((235,230,3),np.uint8);canvas[30:30+resized.shape[0],:resized.shape[1]]=cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas,title.replace("_round2_radiometric",""),(3,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.putText(canvas,f"held-out target {identity}",(3,227),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            cells.append(canvas)
        mosaics.append(np.concatenate(cells,axis=1))
    mosaic=np.concatenate(mosaics,axis=0);cv2.imwrite(str(directory/"holdout_comparison.png"),mosaic)
    cv2.imwrite(str(directory/"holdout_preview.jpg"),cv2.resize(mosaic,(736,376)),[cv2.IMWRITE_JPEG_QUALITY,65])
    print("COMPARISON",json.dumps({k:{x:v[x] for x in ["holdout_ncc","all_target_ncc","holdout_reference_consistency_ssim","sea_above_30_fraction"]} for k,v in result.items()}),flush=True)
    return comparison

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--steps",type=int,default=300);parser.add_argument("--evaluate-only",action="store_true");parser.add_argument("--resume-arms",action="store_true")
    args=parser.parse_args();cv2.setNumThreads(4);torch.set_num_threads(4)
    started=time.time()
    if args.evaluate_only:evaluate();return
    try:
        holdout,fit,initial=fixed_split()
        print("SPLIT",json.loads((ROOT/"evaluation/split.json").read_text()),flush=True)
        summaries={}
        for m in ["legacy","maritime"]:
            saved=ROOT/f"results/{m}/summary.json"
            old=json.loads(saved.read_text()) if args.resume_arms and saved.exists() else []
            complete=len(old)==3 and old[-1].get("round")==3 and all(x["restoration"]["iterations"]==args.steps for x in old[:2])
            if complete:
                summaries[m]=old;print("REUSED_COMPLETED_ARM",m,flush=True)
            else:summaries[m]=run_arm(m,holdout,fit,initial,args.steps)
        comparison=evaluate()
        write_json(ROOT/"experiment_summary.json",dict(summaries=summaries,elapsed_seconds=time.time()-started,evaluation=comparison))
        write_json(ROOT/"STATUS.json",dict(stage="EXPERIMENT_COMPLETE",results_remote_only=True,shutdown_after_completion=True))
    except BaseException:
        write_json(ROOT/"STATUS.json",dict(stage="FAILED_SAVED",traceback=traceback.format_exc(),results_remote_only=True))
        raise
if __name__=="__main__":main()
