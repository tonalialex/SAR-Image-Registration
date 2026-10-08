"""Audit feature origins, target membership, metrics and iterative reference crops."""
import json,sys,cv2,numpy as np
from pathlib import Path
from PIL import Image
from sar_registration.maritime_targets import point_labels,stable_brightness
from sar_registration.geometry import project
from registration_metrics import calculate_registration_metrics
ROOT=Path(__file__).resolve().parent
def main():
    name=sys.argv[1];run=ROOT/'fusion_runs'/name;out=ROOT/'registration_results'/name
    metrics=json.loads((out/'metrics.json').read_text());validation=json.loads((out/'validation_report.json').read_text())
    raw=np.load(run/'patch_coordinates.npz');provenance=np.load(run/'detector_provenance.npz')
    csv=np.loadtxt(out/'matched_points.csv',delimiter=',',skiprows=1,ndmin=2)
    if csv.size==0:csv=np.empty((0,6))
    retained=csv[:,4]>0;src,dst=csv[retained,:2],csv[retained,2:4]
    masks=np.load(run/'maritime_targets'/'target_instances.npz')
    manifest=json.loads((run/'maritime_targets'/'target_instances.json').read_text())
    kinds={int(k):v for k,v in manifest['kinds'].items()}
    a,b=point_labels(src,masks['sensed_labels']),point_labels(dst,masks['reference_labels'])
    def member(points,pool):return all(np.any(np.all(pool==p,axis=1)) for p in points)
    origins=bool(member(src,raw['sensed']) and member(dst,raw['reference']) and
        member(src,provenance['sensed'][:,2:4]) and member(dst,provenance['reference'][:,2:4]))
    same=bool(len(src) and np.all((a>0)&(a==b)))
    h=np.array(metrics['final_transform']);errors=np.linalg.norm(project(src,h)-dst,axis=1)
    weights=csv[retained,6] if csv.shape[1]>6 else np.ones(len(src))
    computed=calculate_registration_metrics(dst,errors,masks['reference_labels'].shape,
                                           source_points=src,weights=weights)
    metric_equal=all((computed[k]==metrics[k] if computed[k] is None else abs(computed[k]-metrics[k])<1e-9)
        for k in ['Nred','RMSEall','RMSEloo','Pquad'])
    latest=json.loads((out/'latest_iterations.json').read_text())['directory']
    previous=None;crop_ok=True;rounds=sorted(d for d in Path(latest).glob('iteration_*') if d.is_dir())
    for directory in rounds:
        record=json.loads((directory/'iteration.json').read_text())
        coords=np.loadtxt(directory/'Xs_coordinates.csv',delimiter=',',skiprows=1,ndmin=2)
        input_h=np.loadtxt(directory/'input_transform.txt')
        crop_ok&=bool(np.allclose(project(coords[:,1:3],input_h),coords[:,3:5],atol=1e-7,rtol=0))
        if previous is not None:crop_ok&=bool(np.allclose(input_h,previous,atol=1e-9,rtol=0))
        previous=np.loadtxt(directory/'output_transform.txt')
        assert (directory/'registration_matches.png').is_file() and (directory/'registration_metrics.txt').is_file()
    counts={kind:int(sum(kinds.get(int(i))==kind for i in b)) for kind in ['ship','shore']}
    report=dict(audit_passed=bool(origins and same and metric_equal and crop_ok and validation['effective']),
        detector_origin_verified=origins,both_ends_on_same_reliable_target=same,
        coordinates_unchanged=True if origins else False,metrics_recomputed=metric_equal,
        iterative_T_to_reference_Xs_verified=crop_ok,attempted_rounds=len(rounds),
        point_count=len(src),points_by_type=counts,reference_brightness_threshold=manifest['brightness_thresholds'][0],
        source_brightness_threshold=manifest['brightness_thresholds'][1],metrics=metrics,validation=validation)
    (out/'target_point_audit.json').write_text(json.dumps(report,indent=2))
    ref=cv2.imread(str(ROOT/'data/ship/reference.jpg'));sensed=cv2.imread(str(ROOT/'data/ship/sensed.jpg'))
    sheet=np.zeros((len(kinds)*220,440,3),np.uint8)
    for row,identity in enumerate(sorted(kinds)):
        for col,(image,labels,points) in enumerate([(ref,masks['reference_labels'],dst),(sensed,masks['sensed_labels'],src)]):
            canvas=image.copy();contours,_=cv2.findContours((labels==identity).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas,contours,-1,(0,255,255),1)
            selected=points[point_labels(points,labels)==identity]
            for p in selected:cv2.circle(canvas,tuple(np.rint(p).astype(int)),3,(0,0,255),1)
            yy,xx=np.where(labels==identity)
            x0=max(0,int(xx.min())-18);x1=min(image.shape[1],int(xx.max())+19)
            y0=max(0,int(yy.min())-18);y1=min(image.shape[0],int(yy.max())+19)
            crop=canvas[y0:y1,x0:x1];scale=min(200/crop.shape[0],200/crop.shape[1])
            crop=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
            sheet[row*220:row*220+crop.shape[0],col*220:col*220+crop.shape[1]]=crop
            cv2.putText(sheet,('R ' if col==0 else 'S ')+kinds[identity]+' '+str(identity)+' N='+str(len(selected)),(col*220,row*220+215),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    cv2.imwrite(str(out/'target_point_closeups.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
    for path in out.rglob('*.png'):
        with Image.open(path) as image:image.verify()
    print(json.dumps({k:report[k] for k in ['audit_passed','detector_origin_verified','both_ends_on_same_reliable_target','metrics_recomputed','iterative_T_to_reference_Xs_verified','attempted_rounds','point_count','points_by_type']},indent=2))
if __name__=='__main__':main()

