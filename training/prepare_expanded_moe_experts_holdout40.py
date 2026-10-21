#!/usr/bin/env python3
"""Frozen-geometry classical expert preparation for expanded Local-MoE.

prepare -> pilot -> compute -> summary. One Slurm task, CPU LAMMPS/DScribe.
No relaxation, MACE inference, new splits, gate fitting or gamma calculation.
Uses the uploaded, hash-pinned classical backend through a Cartesian-frame
adapter. Original geometries and MATPes labels are retained without changes.
The DScribe teacher inputs here are separate from cached TorchSOAP features.
"""
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
import ctypes
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import prepare_expanded_moe_labels as io

EXPERTS = ['ML-BKS', 'Original_BKS', 'Chik_BKS', 'PSO1_BKS', 'PSO2_BKS', 'Tersoff', 'Vashishta']
BUILDER_HASH = '4946b1b2b5738ae3c47f08a37ec0135db4e02a364e391681101648a413b7fd20'
BACKEND_HASH = '204dd4571eb7faa0b5ff17617d1d4ba7699d41bffa7c3e0cfe051df035c71397'
POTENTIAL_HASHES = {
    'Tersoff.pot': '4507541424e6e5f282903615f6740af4d152fb150e8f036bd2e5219381ec1560',
    'Vashishta.pot': 'f2304522fac8b6aeeb25d782917f90e3d3e95414423785bc0b26bb23cd84be63',
}
COUNTS = {'train': 369, 'val': 38, 'test': 105}


def checked(path, expected):
    path = Path(path)
    if io.sha256(path) != expected:
        raise ValueError('Input hash mismatch: ' + str(path))
    return path


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock(run):
    with (run / 'execution.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process is using this expert run')
        yield


def canonical_frame(cell):
    """Row vectors: C_lmp=C@Q, R_lmp=R@Q, F_original=F_lmp@Q.T.

    Reversing one lattice vector first handles a left-handed basis without
    reflecting Cartesian coordinates or changing the periodic lattice.
    """
    c = np.array(cell, dtype=np.float64, copy=True)
    if c.shape != (3, 3) or not np.isfinite(c).all() or abs(np.linalg.det(c)) < 1e-6:
        raise ValueError('Invalid cell')
    basis = np.eye(3)
    if np.linalg.det(c) < 0:
        basis[2, 2] = -1
        c = basis @ c
    ex = c[0] / np.linalg.norm(c[0])
    perpendicular = c[1] - np.dot(c[1], ex) * ex
    ey = perpendicular / np.linalg.norm(perpendicular)
    ez = np.cross(ex, ey)
    q = np.column_stack((ex, ey, ez))
    restricted = c @ q
    if not np.allclose(q.T @ q, np.eye(3), atol=1e-12, rtol=0) or np.linalg.det(q) < 0:
        raise ValueError('Invalid Cartesian rotation')
    if not np.allclose(np.triu(restricted, 1), 0, atol=1e-10, rtol=0):
        raise ValueError('Cell conversion did not give restricted triclinic form')
    restricted[np.triu_indices(3, 1)] = 0
    if np.any(np.diag(restricted) <= 0):
        raise ValueError('Nonpositive restricted cell length')
    if not np.allclose(restricted @ restricted.T, c @ c.T, atol=1e-8, rtol=1e-12):
        raise ValueError('Cell metric changed during rotation')
    return restricted, q, basis


def read_data(dataset, row):
    path = checked(dataset / row['data_file'], row['data_sha256'])
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k].copy() for k in d.files}
    g = io.geometry(data)
    io.basic_qc(g, int(row['n_atoms']), 4608)
    if io.geometry_hash(g) != row['geometry_sha256']:
        raise ValueError('Ordered geometry mismatch: ' + row['record_id'])
    for key in ('record_id', 'split', 'geometry_sha256'):
        if str(data[key].item()) != row[key]:
            raise ValueError('Shard metadata mismatch: ' + key)
    if str(data['teacher'].item()) != io.TEACHER:
        raise ValueError('Not MATPes-r2SCAN labels')
    io.check_labels(float(data['mace_energy_total_eV']), data['mace_forces'], len(g['positions']))
    canonical_frame(g['cell'])
    for key in ('phase', 'interface_center'):
        if key in data and data[key].reshape(-1).shape != (len(g['positions']),):
            raise ValueError('Invalid region metadata')
    return data, g


def prepare(args):
    het, dataset, run = args.het.resolve(), args.dataset.resolve(), args.run.resolve()
    manifest = dataset / 'manifest.csv'
    summary = json.loads((dataset / 'split_summary.json').read_text())
    checked(manifest, summary['manifest_sha256'])
    rows = io.read_csv(manifest)
    if dict(Counter(r['split'] for r in rows)) != COUNTS or summary['unique_configuration_counts'] != COUNTS:
        raise ValueError('Expected frozen 369/38/105 holdout40 dataset')
    ids = [r['record_id'] for r in rows]
    if len(set(ids)) != len(ids) or any(Path(x).name != x or x in ('', '.', '..') for x in ids):
        raise ValueError('Invalid/repeated record IDs')
    for key in ('phase_pair_group', 'split_group', 'geometry_sha256'):
        groups = defaultdict(set)
        for r in rows:
            if not r.get(key):
                raise ValueError('Missing grouping key: ' + key)
            groups[r[key]].add(r['split'])
        if any(len(s) != 1 for s in groups.values()):
            raise ValueError('Cross-split conflict: ' + key)
    info_path = het / 'Local_Expert_Force_Dataset_MATPES_R2SCAN/dataset_info.json'
    info = json.loads(info_path.read_text())
    if info['experts'] != EXPERTS:
        raise ValueError('Historical expert order differs')
    backend = checked(info['core_calculator']['path'], BACKEND_HASH)
    builder = het / 'build_local_expert_force_dataset.py'
    checked(builder, BUILDER_HASH)
    lib = Path(args.lmp_lib or info['lammps_library']['path']).resolve()
    if not lib.is_file():
        raise FileNotFoundError(lib)
    constants = {}
    for node in ast.parse(backend.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'LMP_PYTHON':
                constants['python'] = ast.literal_eval(node.value)
    candidates = [Path(args.lmp_python)] if args.lmp_python else [
        Path(constants['python']), lib.parent.parent / 'python']
    binding = next((p.resolve() for p in candidates if (p / 'lammps/__init__.py').is_file()), None)
    if binding is None:
        raise FileNotFoundError('LAMMPS binding not found; pass --lmp-python directory containing lammps/')
    sources = {'moe_persistent_calculator.py': backend, builder.name: builder,
               Path(__file__).name: Path(__file__).resolve(), Path(io.__file__).name: Path(io.__file__).resolve(),
               'historical_dataset_info.json': info_path}
    for name, expected in POTENTIAL_HASHES.items():
        sources[name] = checked(info['potential_files'][name]['path'], expected)
        if info['potential_files'][name]['sha256'] != expected:
            raise ValueError('Historical potential hash differs')
    orientations = []
    for row in rows:
        _, g = read_data(dataset, row)
        c, q, basis = canonical_frame(g['cell'])
        orientations.append(dict(record_id=row['record_id'], source=row['source'], split=row['split'],
                                 n_atoms=int(row['n_atoms']), rotation_max_abs=float(np.max(np.abs(q-np.eye(3)))),
                                 reversed_lattice_vector=bool(basis[2, 2] < 0)))
    train = [r for r in rows if r['split'] == 'train']
    pilot_ids = []
    for source in sorted(set(r['source'] for r in train)):
        pilot_ids.append(max((r for r in train if r['source'] == source), key=lambda r: int(r['n_atoms']))['record_id'])
    small = min(train, key=lambda r: int(r['n_atoms']))['record_id']
    rotated = max((r for r in orientations if r['split'] == 'train'), key=lambda r: r['rotation_max_abs'])['record_id']
    pilot_ids = list(dict.fromkeys(pilot_ids + [small, rotated]))
    contract = dict(stage='expanded_expert_preparation_v1', dataset=str(dataset),
                    manifest_sha256=io.sha256(manifest), counts=COUNTS, experts=EXPERTS,
                    source_hashes={k: io.sha256(v) for k, v in sources.items()},
                    source_paths={k: str(v) for k, v in sources.items()},
                    historical_backend_sha256=info['core_calculator']['sha256'],
                    current_backend_sha256=BACKEND_HASH, backend_matches_historical=BACKEND_HASH == info['core_calculator']['sha256'],
                    lmp_lib=str(lib), lmp_lib_sha256=io.sha256(lib), lmp_python=str(binding),
                    binding_hashes={str(p.relative_to(binding)): io.sha256(p) for p in sorted((binding/'lammps').rglob('*.py'))},
                    pilot_ids=pilot_ids, numerical_audit_ids=list(dict.fromkeys([small, rotated])),
                    fd_step_A=float(args.fd_step), fd_atol_eV_A=float(args.fd_atol), fd_rtol=float(args.fd_rtol),
                    rotation_force_atol=2e-5, rotation_force_rtol=2e-5,
                    soap=dict(species=['Si', 'O'], r_cut=5.0, n_max=6, l_max=4, feature_dim=392),
                    geometry_policy='original float64 geometry; rotate only LAMMPS input and rotate forces/stress back',
                    reference_policy='copy existing frozen MATPes labels; no classical energy alignment',
                    region_policy='preserve genuine arrays; omit unavailable region labels')
    if run.exists() and any(run.iterdir()):
        raise FileExistsError('Preparation needs a new directory: ' + str(run))
    if run == dataset or run in dataset.parents or dataset in run.parents:
        raise ValueError('Expert outputs must be separate from dataset inputs')
    run.mkdir(parents=True, exist_ok=True)
    for d in ('src_snapshot', 'shards', 'status', 'logs', 'audits'):
        (run/d).mkdir()
    for name, path in sources.items():
        io.snapshot(path, run/'src_snapshot'/name)
    io.snapshot(manifest, run/'input_manifest.csv')
    io.write_csv(run/'orientation_audit.csv', orientations)
    io.write_json(run/'expert_contract.json', contract)
    print('Prepared:', run)
    print('Frozen counts:', COUNTS)
    print('Current calculator matches historical file:', contract['backend_matches_historical'])
    print('Records needing Cartesian rotation:', sum(r['rotation_max_abs'] > 1e-10 for r in orientations))
    print('Pilot IDs:', ', '.join(pilot_ids))
    print('LAMMPS binding:', binding)
    print('No calculator loaded; no structures modified; no expert forces computed.')


def load_run(run):
    run = run.resolve()
    c = json.loads((run/'expert_contract.json').read_text())
    checked(run/'input_manifest.csv', c['manifest_sha256'])
    checked(Path(c['dataset'])/'manifest.csv', c['manifest_sha256'])
    for name, expected in c['source_hashes'].items():
        checked(run/'src_snapshot'/name, expected)
    checked(Path(__file__), c['source_hashes'][Path(__file__).name])
    checked(Path(io.__file__), c['source_hashes'][Path(io.__file__).name])
    return c, io.read_csv(run/'input_manifest.csv'), io.sha256(run/'expert_contract.json')


def libraries_loaded(lib):
    """Refuse silent use of a different liblammps by the Python wrapper."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[-1].startswith('/') and 'liblammps' in Path(parts[-1]).name:
            p = Path(parts[-1])
            if not p.is_file() or not os.path.samefile(p, lib):
                raise RuntimeError('Unexpected loaded LAMMPS library: ' + str(p))
            paths.add(str(p.resolve()))
    if not paths:
        raise RuntimeError('Cannot verify loaded LAMMPS library')
    return sorted(paths)


def startup(run, c):
    if int(os.environ.get('SLURM_NTASKS', '1')) != 1:
        raise RuntimeError('Use exactly ONE Slurm task: the MPI backend uses its default communicator')
    os.environ['MPICH_GPU_SUPPORT_ENABLED'] = '0'
    checked(c['lmp_lib'], c['lmp_lib_sha256'])
    for name, h in c['binding_hashes'].items():
        checked(Path(c['lmp_python'])/name, h)
    ctypes.CDLL(c['lmp_lib'], ctypes.RTLD_GLOBAL)
    sys.path.insert(0, c['lmp_python'])
    import lammps
    if not Path(lammps.__file__).resolve().is_relative_to(Path(c['lmp_python'])):
        raise RuntimeError('Imported unexpected LAMMPS Python package')
    import torch
    torch.set_num_threads(1)
    builder = import_file(run/'src_snapshot/build_local_expert_force_dataset.py', 'expanded_expert_builder')
    backend = import_file(run/'src_snapshot/moe_persistent_calculator.py', 'expanded_classical_backend')
    if list(backend.EXPERT_SETUP) != EXPERTS or builder.EXPERT_NAMES != EXPERTS:
        raise ValueError('Expert order mismatch')
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'ase', 'dscribe', 'torch')}
    runtime = dict(versions=versions, lammps_python=str(Path(lammps.__file__).resolve()),
                   omp_threads=os.environ.get('OMP_NUM_THREADS', ''), mkl_threads=os.environ.get('MKL_NUM_THREADS', ''))
    io.publish_contract(run/'runtime.json', runtime)
    descriptor = builder.LocalSOAPDescriptor(5.0, 6, 4, 1, 390)
    return backend, descriptor


def run_one_expert(g, name, backend, c, run):
    restricted, q, basis = canonical_frame(g['cell'])
    transformed = dict(g, positions=g['positions']@q, cell=restricted)
    atoms = io.as_atoms(transformed)
    atoms.wrap(eps=1e-12)
    engine = None
    try:
        engine = backend.PersistentLammps(name, atoms, str(run/'src_snapshot'), c['lmp_lib'], c['lmp_python'])
        # Make total PE semantics explicit; backend returns PE divided by N.
        engine.lmp.command('thermo_modify norm no')
        energy_per_atom, forces, stress = engine.compute(atoms)
        libraries_loaded(c['lmp_lib'])
        n = len(atoms)
        ids = np.array(engine.lmp.gather_atoms('id', 0, 1), dtype=np.int64).reshape(-1)
        types = np.array(engine.lmp.gather_atoms('type', 0, 1), dtype=np.int64).reshape(-1)
        if not np.array_equal(ids, np.arange(1,n+1)) or not np.array_equal(types, np.where(g['atomic_numbers']==14,1,2)):
            raise ValueError('LAMMPS atom order/species changed')
        forces = np.asarray(forces, dtype=np.float64)
        stress = np.asarray(stress, dtype=np.float64)
        if forces.shape != (n,3) or stress.shape != (6,) or not np.isfinite(stress).all():
            raise ValueError('Invalid expert force/stress output')
        io.check_labels(float(energy_per_atom)*n, forces, n)
        xx,yy,zz,yz,xz,xy = stress
        s = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
        original_s = q @ s @ q.T
        original_voigt = original_s[[0,1,2,1,0,0],[0,1,2,2,2,1]]
        return float(energy_per_atom)*n, forces@q.T, original_voigt
    finally:
        if engine is not None:
            engine.close()


def numerical_audit(g, energies, forces, backend, c, run):
    # Probe one Si and one O Cartesian degree of freedom in the original frame.
    dofs = [(int(np.flatnonzero(g['atomic_numbers']==14)[0]), 0),
            (int(np.flatnonzero(g['atomic_numbers']==8)[0]), 1)]
    angle = 0.371
    rot = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    rotated = dict(g, cell=g['cell']@rot, positions=g['positions']@rot)
    result = []
    for k, name in enumerate(EXPERTS):
        e_rot, f_rot, _ = run_one_expert(rotated, name, backend, c, run)
        if not np.isclose(e_rot, energies[k], atol=1e-5, rtol=1e-9) or not np.allclose(
                f_rot@rot.T, forces[k], atol=c['rotation_force_atol'], rtol=c['rotation_force_rtol']):
            raise ValueError('Rotation covariance failed: ' + name)
        for atom, axis in dofs:
            epm = []
            for sign in (1, -1):
                positions = g['positions'].copy()
                positions[atom,axis] += sign*c['fd_step_A']
                epm.append(run_one_expert(dict(g,positions=positions), name, backend,c,run)[0])
            fd = -(epm[0]-epm[1])/(2*c['fd_step_A'])
            actual = float(forces[k,atom,axis])
            error = abs(fd-actual)
            limit = c['fd_atol_eV_A']+c['fd_rtol']*max(abs(fd),abs(actual))
            result.append(dict(expert=name, atom_index=atom, axis=axis, force=actual,
                               finite_difference_force=fd, abs_error=error, tolerance=limit, passed=error<=limit,
                               rotation_max_force_error=float(np.max(np.abs(f_rot@rot.T-forces[k])))))
    return result


def cached(run, row, contract_hash, deep=True):
    rid = row['record_id']; marker=run/'status'/(rid+'.json'); shard=run/'shards'/(rid+'.npz')
    if not marker.is_file():
        return False, 'PENDING'
    try:
        s=json.loads(marker.read_text())
        if s['status']!='PASS': return False,s['status']
        if s['contract_sha256']!=contract_hash or io.sha256(shard)!=s['shard_sha256']:
            return False,'INVALID_HASH'
        if deep:
            with np.load(shard,allow_pickle=False) as d:
                n=int(row['n_atoms'])
                if d['expert_forces'].shape!=(7,n,3) or d['X_local'].shape!=(n,392): return False,'INVALID_SHAPE'
                if d['expert_names'].tolist()!=EXPERTS: return False,'INVALID_EXPERT_ORDER'
                if io.geometry_hash(io.geometry(d))!=row['geometry_sha256']: return False,'INVALID_GEOMETRY'
                if str(d['split'].item())!=row['split'] or str(d['record_id'].item())!=rid: return False,'INVALID_ID'
                for key in ('X_local','expert_forces','expert_energy_total_raw','expert_stress','mace_forces','mace_energy_total_eV'):
                    if not np.isfinite(d[key]).all(): return False,'NONFINITE'
        return True,'PASS'
    except Exception:
        return False,'INVALID_CACHE'


def compute(args, pilot=False):
    run=args.run.resolve()
    with lock(run):
        c,rows,ch=load_run(run)
        if not pilot:
            passed=json.loads((run/'pilot_summary.json').read_text())
            if passed['status']!='PASS' or passed['contract_sha256']!=ch:
                raise ValueError('Numerical pilot has not passed')
            for rid in c['pilot_ids']:
                row=next(r for r in rows if r['record_id']==rid)
                if not cached(run,row,ch)[0]: raise ValueError('Pilot cache no longer valid: '+rid)
            for rid in c['numerical_audit_ids']:
                checked(run/'audits'/(rid+'.json'),passed['audit_hashes'][rid])
        selected=[r for r in rows if not pilot or r['record_id'] in c['pilot_ids']]
        backend,descriptor=startup(run,c)
        started=time.monotonic();failures=0;processed=0;skipped=0
        if pilot: io.write_json(run/'pilot_summary.json',dict(status='RUNNING',contract_sha256=ch))
        for index,row in enumerate(selected,1):
            rid=row['record_id'];data,g=read_data(Path(c['dataset']),row)
            good,_=cached(run,row,ch)
            audit_file=run/'audits'/(rid+'.json')
            if good:
                if pilot and rid in c['numerical_audit_ids']:
                    good=audit_file.is_file() and json.loads(audit_file.read_text()).get('status')=='PASS'
                if good:
                    skipped+=1;continue
            if args.max_hours>0 and time.monotonic()-started>args.max_hours*3600:
                print('Paused between records; rerun the same command to resume.',flush=True);break
            marker=run/'status'/(rid+'.json')
            status=dict(record_id=rid,contract_sha256=ch,status='RUNNING',started=io.now())
            io.write_json(marker,status);start=time.monotonic()
            try:
                with (run/'logs'/(rid+'.log')).open('a',buffering=1) as log, redirect_stdout(log):
                    x,_=descriptor.compute(io.as_atoms(g),np.where(g['atomic_numbers']==14,'Si','O'))
                    if x.shape!=(int(row['n_atoms']),392) or not np.isfinite(x).all():
                        raise ValueError('Invalid DScribe features')
                    expected=np.stack([g['atomic_numbers']==14,g['atomic_numbers']==8],axis=1)
                    if not np.array_equal(x[:,-2:],expected): raise ValueError('Species feature mismatch')
                    values=[run_one_expert(g,name,backend,c,run) for name in EXPERTS]
                    energies=np.array([v[0] for v in values]); forces=np.stack([v[1] for v in values]); stresses=np.stack([v[2] for v in values])
                    if pilot and rid in c['numerical_audit_ids']:
                        audits=numerical_audit(g,energies,forces,backend,c,run)
                        passed=all(a['passed'] for a in audits)
                        io.write_json(audit_file,dict(status='PASS' if passed else 'FAIL',contract_sha256=ch,checks=audits))
                        if not passed: raise ValueError('Force/energy finite-difference audit failed; see '+str(audit_file))
                # Preserve all frozen reference labels and original-frame geometry.
                shard=dict(data,**g,X_local=x.astype(np.float32),expert_names=np.array(EXPERTS),
                           expert_forces=forces,expert_energy_total_raw=energies,
                           expert_energy_per_atom_raw=energies/int(row['n_atoms']),expert_stress=stresses,
                           has_region_metadata=np.array('phase' in data and 'interface_center' in data),
                           expert_contract_sha256=np.array(ch))
                target=run/'shards'/(rid+'.npz');io.write_npz(target,shard)
                status.update(status='PASS',shard_sha256=io.sha256(target),finished=io.now(),seconds=time.monotonic()-start)
                io.write_json(marker,status)
                if not cached(run,row,ch)[0]: raise ValueError('Written shard did not verify')
                processed+=1
                print(f"[{index}/{len(selected)}] {rid} {row['source']} N={row['n_atoms']} PASS {time.monotonic()-start:.2f}s",flush=True)
            except Exception as exc:
                failures+=1;status.update(status='FAIL',error=str(exc),traceback=traceback.format_exc(),finished=io.now())
                io.write_json(marker,status)
                print(f'{rid} FAIL {type(exc).__name__}: {exc}',flush=True)
                break
        if pilot:
            ok=all(cached(run,r,ch)[0] for r in selected)
            hashes={}
            for rid in c['numerical_audit_ids']:
                p=run/'audits'/(rid+'.json')
                valid=p.is_file() and json.loads(p.read_text()).get('status')=='PASS' and json.loads(p.read_text()).get('contract_sha256')==ch
                ok=ok and valid
                if valid: hashes[rid]=io.sha256(p)
            io.write_json(run/'pilot_summary.json',dict(status='PASS' if ok else 'INCOMPLETE',contract_sha256=ch,
                pilot_ids=c['pilot_ids'],audit_hashes=hashes,finished=io.now()))
            print('Numerical pilot:', 'PASS' if ok else 'INCOMPLETE')
        print(json.dumps(dict(processed=processed,verified_existing_pass=skipped,failures=failures)))
        summarize(run)
        return 2 if failures else (3 if pilot and not ok else 0)


def summarize(run):
    c,rows,ch=load_run(run)
    report=[];complete=[]
    for row in rows:
        read_data(Path(c['dataset']),row)
        valid,status=cached(run,row,ch)
        report.append(dict(row,status=status))
        if valid:
            complete.append(dict(row,expert_file=str(run/'shards'/(row['record_id']+'.npz')),
                                 expert_sha256=io.sha256(run/'shards'/(row['record_id']+'.npz'))))
    io.write_csv(run/'expert_status.csv',report)
    io.write_csv(run/'completed_expert_manifest.csv',complete,fields=list(rows[0])+['expert_file','expert_sha256'])
    for source in sorted(set(r['source'] for r in rows)):
        print(source,dict(Counter(r['status'] for r in report if r['source']==source)))
    all_pass=len(complete)==len(rows)
    io.write_json(run/'expert_summary.json',dict(status='COMPLETE' if all_pass else 'COMPLETED_SUBSET',
        counts=dict(Counter(r['status'] for r in report)),completed_by_split=dict(Counter(r['split'] for r in complete)),
        contract_sha256=ch,manifest_sha256=io.sha256(run/'completed_expert_manifest.csv'),finished=io.now()))
    print('Expert manifest:',run/'completed_expert_manifest.csv')
    print('No gate training or conservative-model predictions performed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','pilot','compute','summary'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--het',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--lmp-lib')
    p.add_argument('--lmp-python')
    p.add_argument('--max-hours',type=float,default=3.5)
    p.add_argument('--fd-step',type=float,default=1e-4,
                   help='Finite-difference displacement used by the pilot (A).')
    p.add_argument('--fd-atol',type=float,default=8e-3,
                   help='Absolute force tolerance for the pilot (eV/A).')
    p.add_argument('--fd-rtol',type=float,default=1e-2,
                   help='Relative force tolerance for the pilot.')
    a=p.parse_args();a.run=a.run.resolve()
    if not np.isfinite(a.max_hours) or a.max_hours<0: p.error('--max-hours must be finite and nonnegative')
    if not np.isfinite(a.fd_step) or a.fd_step <= 0: p.error('--fd-step must be positive')
    if not np.isfinite(a.fd_atol) or a.fd_atol <= 0: p.error('--fd-atol must be positive')
    if not np.isfinite(a.fd_rtol) or a.fd_rtol < 0: p.error('--fd-rtol must be nonnegative')
    if a.command=='prepare':
        if a.het is None or a.dataset is None: p.error('prepare requires --het and --dataset')
        prepare(a);return 0
    if a.command=='summary':
        with lock(a.run): summarize(a.run)
        return 0
    return compute(a,pilot=a.command=='pilot')


if __name__=='__main__':
    raise SystemExit(main())
