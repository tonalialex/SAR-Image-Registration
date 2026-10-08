"""Use a historical metric baseline only for the identical image inputs."""
import hashlib,json
from pathlib import Path
from registration_metrics import RMSELOO_METHOD
def input_fingerprint(paths):
    labels=['reference','sensed']+[f'restored_{i}' for i in range(1,len(paths)-1)]
    return {label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for label,path in zip(labels,paths)}
def load_comparable_baseline(root,paths):
    directory=Path(root)/'baseline_reference';manifest=directory/'input_sha256.json';metrics=directory/'metrics.json'
    comparable=bool(manifest.is_file() and metrics.is_file() and json.loads(manifest.read_text())==input_fingerprint(paths))
    baseline = json.loads(metrics.read_text()) if comparable else None
    return baseline if baseline is not None and baseline.get('RMSEloo_method') == RMSELOO_METHOD else None

