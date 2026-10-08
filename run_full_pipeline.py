#!/usr/bin/env python3
"""One command: two pair-specific restorations, then three-layer registration."""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    temporary.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def discover(directory):
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'})
    groups = {}
    for p in files:
        m = re.fullmatch(r'(.+)_(active|passive)(?:_PGA\d+)?', p.stem, re.IGNORECASE)
        if not m: raise ValueError(f'Unrecognized image filename: {p.name}')
        key, role = m[1], m[2].lower()
        if role in groups.setdefault(key, {}): raise ValueError(f'Duplicate {role}: {key}')
        groups[key][role] = p.resolve()
    pairs = []
    for key, roles in sorted(groups.items()):
        if set(roles) != {'active','passive'}: raise ValueError(f'Incomplete pair: {key}')
        pairs.append((key, roles['active'], roles['passive']))
    if not pairs: raise ValueError('No image pairs found')
    return pairs

def run(command, cwd, logfile, env, record):
    record.append(dict(command=command, cwd=str(cwd), log=str(logfile)))
    print('RUN '+ ' '.join(map(str,command)), flush=True)
    with logfile.open('w', encoding='utf-8') as f:
        code = subprocess.run(command, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if code: raise RuntimeError(f'Exit {code}; see {logfile}')

def process(key, reference, sensed, out, args):
    started = time.time()
    if out.exists() and args.continue_existing:
        previous = json.loads((out/'pair_summary.json').read_text())
        if previous['inputs_sha256'] != dict(reference=digest(reference),sensed=digest(sensed)):
            raise ValueError(f'Existing input hashes differ: {key}')
        if previous['status'] in {'accepted','rejected_retained_T0'}:
            commands = previous['commands']
            for command, flag, expected in [(commands[0]['command'],'--iterations',str(args.restoration_steps)),
                                             (commands[0]['command'],'--seed',str(args.restoration_seed)),
                                             (commands[1]['command'],'--contrastive-epochs',str(args.contrastive_epochs))]:
                if command[command.index(flag)+1] != expected: raise ValueError(f'Existing configuration differs: {key}')
            if not (out/'results/metrics.json').is_file(): raise FileNotFoundError(out/'results/metrics.json')
            saved_metrics = json.loads((out/'results/metrics.json').read_text())
            if saved_metrics.get('RMSEloo_method') != 'leave_one_out_affine_prediction':
                raise ValueError(f'Existing report uses the old RMSEloo definition; rerun in a new output directory: {key}')
            print(f'REUSE completed {key}',flush=True)
            return previous
        preserved=out.parent/'interrupted_attempts'/(key+'_'+str(time.time_ns()))
        preserved.parent.mkdir(exist_ok=True)
        out.rename(preserved)
        print(f'PRESERVED incomplete attempt: {preserved}',flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    summary = dict(pair=key, reference=str(reference), sensed=str(sensed), status='running',
                   inputs_sha256=dict(reference=digest(reference), sensed=digest(sensed)), commands=[])
    save(out/'pair_summary.json', summary)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    phase='restoration'
    try:
        rest = ROOT/'restoration'
        run([sys.executable, str(rest/'run_maritime_restoration.py'), '--reference',str(reference),'--sensed',str(sensed),
             '--output-dir',str(out/'restoration'),'--iterations',str(args.restoration_steps),'--device',args.device,
             '--seed',str(args.restoration_seed),'--full-resolution-retry','--keep-previous-geometry'], rest, out/'logs/restoration.log', env, summary['commands'])
        r1 = out/'restoration/restoration_after_round1/restored_sensed_radiometric.png'
        r2 = out/'restoration/restoration_after_round2/restored_sensed_radiometric.png'
        from PIL import Image
        with Image.open(sensed) as im: size = im.size
        for p in [r1,r2]:
            with Image.open(p) as im:
                if im.size != size: raise ValueError(f'Grid mismatch: {p}')
                im.verify()
        summary['restorations'] = [str(r1),str(r2)]
        phase='registration'
        reg = ROOT/'registration'
        run([sys.executable,str(reg/'run_registration.py'),'--original',str(sensed),'--reference',str(reference),
             '--round1',str(r1),'--round2',str(r2),'--dataset',key,'--output-root',str(out/'registration'),
             '--device',args.device,'--batch-size','64','--contrastive-epochs',str(args.contrastive_epochs),
             '--steps-per-epoch','12','--seed','20260709','--matching-iterations','4','--layer-confidences','1,1,1',
             '--min-final-matches','6','--max-final-matches','24','--relaxed-spatial',
             '--warm-start',str(ROOT/'models/simclr_initial.pt')], reg, out/'logs/registration.log',env,summary['commands'])
        results = out/'registration/registration_results'/key
        shutil.copytree(results, out/'results')
        summary['metrics'] = json.loads((results/'metrics.json').read_text())
        summary['validation'] = json.loads((results/'validation_report.json').read_text())
        summary['status'] = 'accepted' if summary['validation']['effective'] else 'rejected_retained_T0'
        summary['result_directory'] = str(out/'results')
        required = ['metrics.json','registration_metrics.txt','final_registration_matches.png','registered_sensed.png',
                    'registration_overlay.png','final_transform.txt','matched_points.csv','validation_report.json']
        for name in required:
            if not (results/name).is_file(): raise FileNotFoundError(results/name)
        for p in results.rglob('*.png'):
            with Image.open(p) as im: im.verify()
    except Exception as exc:
        summary['status'] = 'execution_failed'
        summary['error'] = repr(exc)
        summary['stage_failed'] = phase
        summary['restorations'] = [str(p) for p in [out/'restoration/restoration_after_round1/restored_sensed_radiometric.png',
                                                   out/'restoration/restoration_after_round2/restored_sensed_radiometric.png'] if p.is_file()]
        log=out/'logs'/f'{phase}.log'
        if log.is_file():
            lines=log.read_text(errors='replace').splitlines()
            summary['failure_reason']=next((line for line in reversed(lines) if line.strip()),repr(exc))
        summary['traceback'] = traceback.format_exc()
        (out/'results').mkdir(exist_ok=True)
        failure_metrics=dict(status='execution_failed',stage_failed=phase,Nred=None,RMSEall=None,RMSEloo=None,Pquad=None,
                             metric_note='No final accepted correspondence population; metrics unavailable',
                             reason=summary.get('failure_reason',repr(exc)))
        save(out/'results/metrics.json',failure_metrics)
        save(out/'results/failure_report.json',summary)
        diagnostic=sorted((out/'restoration').glob('round*_sarsift*/sarsift_connecting_lines.png'))
        if diagnostic:shutil.copy2(diagnostic[-1],out/'results/coarse_diagnostic_matches.png')
        summary['result_directory']=str(out/'results')
        print(f'FAILED {key}: {exc}', flush=True)
    summary['elapsed_seconds'] = time.time()-started
    save(out/'pair_summary.json',summary)
    print(f"DONE {key}: {summary['status']} ({summary['elapsed_seconds']:.1f}s)", flush=True)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir',type=Path)
    inputs.add_argument('--sensed',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--restoration-steps',type=int,default=300)
    p.add_argument('--restoration-seed',type=int,default=2026)
    p.add_argument('--contrastive-epochs',type=int,default=100)
    p.add_argument('--expected-pairs',type=int)
    p.add_argument('--workers',type=int,default=1,choices=range(1,4),help='Independent pair processes; 1 to 3')
    p.add_argument('--continue-existing',action='store_true',help='Reuse complete verified pairs; preserve and retry incomplete attempts')
    a = p.parse_args()
    if a.restoration_steps<1 or a.contrastive_epochs<1: p.error('Training counts must be positive')
    if a.input_dir:
        pairs = discover(a.input_dir.resolve())
    else:
        if a.reference is None: p.error('--reference is required with --sensed')
        pairs = [('single_pair',a.reference.resolve(),a.sensed.resolve())]
    if a.expected_pairs is not None and len(pairs)!=a.expected_pairs: raise ValueError(f'Expected {a.expected_pairs} pairs, found {len(pairs)}')
    output = a.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=a.continue_existing)
    save(output/'input_pairs.json',[dict(pair=k,reference=str(r),sensed=str(s)) for k,r,s in pairs])
    summaries=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(process,k,r,s,output/k,a) for k,r,s in pairs]
        for future in as_completed(futures):
            summaries.append(future.result())
            summaries.sort(key=lambda s:s['pair'])
            save(output/'batch_summary.json',dict(configuration=vars(a),completed=len(summaries),total=len(pairs),pairs=summaries))
    keys=['pair','status','Nred','RMSEall','RMSEloo','Pquad','predictive_loo_rmse','elapsed_seconds','error']
    with (output/'batch_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for s in summaries:
            row={k:s.get(k) for k in keys};row.update({k:s.get('metrics',{}).get(k) for k in keys[2:6]})
            row['predictive_loo_rmse']=s.get('validation',{}).get('predictive_loo_rmse');writer.writerow(row)
    return 1 if any(s['status']=='execution_failed' for s in summaries) else 0

if __name__=='__main__': raise SystemExit(main())
