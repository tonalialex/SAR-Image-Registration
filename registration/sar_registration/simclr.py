"""Reference-only multiscale SimCLR v1 for local SAR descriptors.

AdaSSIR III-A: train around reference keypoints; map sensed coordinates
through T before reference-image inference crops. No sensed-image patches.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sar_registration.geometry import project

PAPER_SIZES = tuple(range(56, 129, 8))


@dataclass
class LearningConfig:
    epochs: int = 100
    batch_size: int = 128
    workers: int = 0
    lr: float = .0003
    temperature: float = .15
    descriptor_dim: int = 256
    projection_dim: int = 128
    patch_size: int = 64
    seed: int = 20260709
    device: str = 'cuda'
    resume: bool = False
    steps_per_epoch: int = 0
    rotation_degrees: float = 5.
    warm_start: str | None = None


def crop_reference(reference, center, size, output=64):
    """Subpixel center, strict full-window bounds, then uniform-size resize."""
    x, y = map(float, center)
    radius = (size - 1) / 2
    if not (np.isfinite([x, y]).all() and radius <= x <= reference.shape[1]-1-radius
            and radius <= y <= reference.shape[0]-1-radius):
        raise ValueError(f'Patch {size} at {(x, y)} exceeds reference bounds')
    patch = cv2.getRectSubPix(reference, (int(size), int(size)), (x, y))
    return cv2.resize(patch, (output, output), interpolation=cv2.INTER_AREA)


def valid_centers(points, shape, size=128):
    points = np.asarray(points).reshape(-1, 2)
    r = (size-1)/2
    return (np.isfinite(points).all(1) & (points[:, 0] >= r) & (points[:, 1] >= r)
            & (points[:, 0] <= shape[1]-1-r) & (points[:, 1] <= shape[0]-1-r))


class ReferenceInstances(Dataset):
    def __init__(self, reference, points, output=64,rotation_degrees=5.):
        self.reference, self.points, self.output = reference, points, output
        self.rotation_degrees=rotation_degrees

    def __len__(self):
        return len(self.points)

    def view(self, point, size):
        patch = crop_reference(self.reference, point, size, self.output).astype(np.float32)/255.
        # A pair shares its paper crop scale. Strong independent D4 transforms
        # erase the spatial layout needed to distinguish adjacent keypoints.
        rotation = cv2.getRotationMatrix2D(((self.output-1)/2,)*2,
                                          random.uniform(-self.rotation_degrees, self.rotation_degrees), 1.)
        patch = cv2.warpAffine(patch, rotation, (self.output, self.output),
                               borderMode=cv2.BORDER_REFLECT_101)
        patch = np.clip(patch, 0, 1) ** random.uniform(.85, 1.2)
        patch *= random.uniform(.9, 1.1)
        if random.random() < .5:
            patch *= np.random.gamma(40., 1/40., patch.shape).astype(np.float32)
        if random.random() < .5:
            patch = cv2.GaussianBlur(patch, (5, 5), random.uniform(.1, .7))
        return torch.from_numpy(np.clip(patch, 0, 1).copy()).unsqueeze(0)

    def __getitem__(self, index):
        size = random.choice(PAPER_SIZES)
        return self.view(self.points[index], size), self.view(self.points[index], size)


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class SARSimCLR(nn.Module):
    """Residual encoder with global, central and 4x4 spatial descriptors.

    h is used for matching; g(h) is used only by the SimCLR objective.
    """
    def __init__(self, descriptor_dim=256, projection_dim=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(Residual(32, 32), Residual(32, 64, 2),
            Residual(64, 64), Residual(64, 128, 2), Residual(128, 128),
            Residual(128, 192, 2), Residual(192, 192))
        # Retain central and 4x4 layout as well as broad context. Matching is
        # sensitive to translations instead of being dominated by global pools.
        self.descriptor = nn.Linear(192*18, descriptor_dim)
        self.projector = nn.Sequential(nn.Linear(descriptor_dim, descriptor_dim, bias=False),
            nn.BatchNorm1d(descriptor_dim), nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, projection_dim))
        self.register_buffer('sobel', torch.tensor([[[[-1.,0,1],[-2,0,2],[-1,0,1]]],
                                                   [[[-1.,-2,-1],[0,0,0],[1,2,1]]]]) / 8)

    def encode(self, x):
        log = torch.log1p(10*x)/math.log(11)
        normalized = (log-log.mean((2,3), keepdim=True))/(log.std((2,3), keepdim=True)+.05)
        gradients = F.conv2d(log, self.sobel, padding=1)
        y = self.blocks(self.stem(torch.cat((normalized, gradients), dim=1)))
        central = y[:, :, 2:6, 2:6].mean((2,3))
        y = torch.cat((F.adaptive_avg_pool2d(y, 1).flatten(1), central,
                       F.adaptive_avg_pool2d(y, 4).flatten(1)), dim=1)
        return self.descriptor(y)

    def forward(self, x):
        return F.normalize(self.projector(self.encode(x)), dim=1)


def nt_xent(first, second, temperature):
    """Exact symmetric 2N SimCLR v1 loss; mask self, retain positive in denominator."""
    n = len(first)
    if n < 2:
        raise ValueError('NT-Xent needs >=2 distinct instances per batch')
    z = F.normalize(torch.cat((first, second)).float(), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    labels = (torch.arange(2*n, device=z.device)+n) % (2*n)
    return F.cross_entropy(logits, labels)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    cv2.setNumThreads(1)


def train(reference, points, directory: Path, config: LearningConfig):
    if len(points) < 4:
        raise RuntimeError('Fewer than four reference training keypoints with valid 128px crops')
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install CUDA-enabled PyTorch')
    model = SARSimCLR(config.descriptor_dim, config.projection_dim).to(device)
    if config.warm_start and not config.resume:
        initial=torch.load(config.warm_start,map_location='cpu',weights_only=False)
        model.load_state_dict(initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    batch = min(config.batch_size, len(points))
    generator = torch.Generator().manual_seed(config.seed)
    if config.steps_per_epoch:
        class UniqueInstanceBatches:
            def __len__(self):
                return config.steps_per_epoch
            def __iter__(self):
                for _ in range(config.steps_per_epoch):
                    yield torch.randperm(len(points),generator=generator)[:batch].tolist()
        loader=DataLoader(ReferenceInstances(reference,points,config.patch_size,config.rotation_degrees),
            batch_sampler=UniqueInstanceBatches(),num_workers=config.workers,
            pin_memory=amp,worker_init_fn=seed_worker,
            **({'multiprocessing_context':'spawn'} if config.workers else {}))
    else:
        loader = DataLoader(ReferenceInstances(reference,points,config.patch_size,config.rotation_degrees),
            batch_size=batch,shuffle=True,drop_last=True,num_workers=config.workers,
            pin_memory=amp,worker_init_fn=seed_worker,generator=generator,
            **({'multiprocessing_context':'spawn'} if config.workers else {}))
    fingerprint = hashlib.sha256(reference.tobytes()+points.tobytes()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/'simclr_last.pt'
    history, start = [], 0
    metadata = {k:v for k,v in asdict(config).items() if k not in ('resume', 'device', 'workers')}
    metadata['architecture'] = 'center_layout_simclr_v2'
    if config.warm_start:
        metadata['warm_start_sha256']=hashlib.sha256(Path(config.warm_start).read_bytes()).hexdigest()
    if config.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f'--resume requested but checkpoint missing: {checkpoint}')
        # Only load this locally generated checkpoint, never an untrusted model.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['fingerprint'] != fingerprint or state['config'] != metadata:
            raise ValueError('Checkpoint inputs/config mismatch; use a new dataset name')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start, history = state['epoch']+1, state['history']
        random.setstate(state['random']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']); generator.set_state(state['loader'])
        if amp and state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
    elif checkpoint.exists():
        raise FileExistsError(f'{checkpoint} exists; use --resume or another --dataset')
    for epoch in range(start, config.epochs):
        model.train()
        total, count = 0., 0
        for a, b in loader:
            a, b = a.to(device, non_blocking=True), b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                h = model.encode(torch.cat((a, b)))
                z = F.normalize(model.projector(h), dim=1)
                # Directly train the pre-projection descriptor used at test time;
                # projector-only contrastive loss can hide poor localization h.
                loss = nt_xent(z[:len(a)], z[len(a):], config.temperature)
                loss = loss + .25*nt_xent(h[:len(a)], h[len(a):], config.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite contrastive loss; checkpoint preserved')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update()
            total += loss.item()*len(a); count += len(a)
        scheduler.step()
        history.append({'epoch':epoch+1, 'loss':total/count, 'instances':count})
        print(f'SimCLR {epoch+1}/{config.epochs} loss={total/count:.5f}', flush=True)
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
            fingerprint=fingerprint, config=metadata, history=history,
            random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
            loader=generator.get_state(), cuda=torch.cuda.get_rng_state_all() if amp else None)
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(checkpoint)
        (directory/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return model.eval()


@torch.inference_mode()
def describe(model, reference, centers, batch_size=128):
    """Largest (128px) reference crops only, exactly as paper test sampling."""
    device = next(model.parameters()).device
    descriptors = []
    for start in range(0, len(centers), batch_size):
        patches = np.stack([crop_reference(reference, p, 128) for p in centers[start:start+batch_size]])
        x = torch.from_numpy(patches.astype(np.float32)/255.).unsqueeze(1).to(device)
        descriptors.append(F.normalize(model.encode(x), dim=1).cpu().numpy())
    return np.concatenate(descriptors) if descriptors else np.empty((0, model.descriptor.out_features), np.float32)
