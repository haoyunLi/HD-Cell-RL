#!/usr/bin/env python3
"""Switch an explicit dataset entry point, retaining a recoverable old directory."""
import argparse
import json
from pathlib import Path


def activate(dataset,active,condition,backup_tag):
    dataset=Path(dataset).resolve(strict=True);active=Path(active).absolute()
    done=json.loads((dataset/'completion.json').read_text())
    if not done['complete'] or not done['whole_section'] or not done['covered_bins_unchanged']:
        raise ValueError('paired dataset is not complete and validated')
    if condition not in {'no_background','with_background'}:
        raise ValueError('unknown paired condition')
    target=dataset/condition
    if not (target/'filtered_feature_bc_matrix/matrix.mtx.gz').is_file():raise FileNotFoundError(target)
    if active.name!='colorectal_nucleus_based_multi_owner_v2' or active.parent.resolve()!=dataset.parent:
        raise ValueError('activation is restricted to the explicit colorectal pseudo-HD entry point')
    if not backup_tag or '/' in backup_tag or '..' in backup_tag:raise ValueError('invalid backup tag')
    backup=active.with_name(active.name+'.backup_'+backup_tag)
    if backup.exists() or backup.is_symlink():raise FileExistsError(backup)
    if not active.exists() and not active.is_symlink():raise FileNotFoundError(active)
    active.rename(backup)
    try:active.symlink_to(target,target_is_directory=True)
    except BaseException:
        backup.rename(active)
        raise
    receipt=dict(active=str(active),target=str(target),condition=condition,backup=str(backup),
                 deleted_files=0,dataset_role='experimental_control',biological_realism_validated=False,
                 paired_integrity_validated=True,
                 note='Old data preserved; use backup path to reproduce historical datasets. Realism thresholds are not all passed.')
    (dataset/'activation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (target/'EXPERIMENTAL_CONTROL.md').write_text(
        '# Experimental control dataset\n\n'
        f'Active condition: `{condition}`.\n\n'
        'This dataset passed structural and paired-count integrity checks, but did not pass all '
        'predefined biological-realism thresholds. It is an experimental control, not a validated '
        'replacement for real Visium HD data.\n\n'
        f'Independent expression reference: `{dataset / "reference/reference_counts.npz"}`. '
        'Do not reuse the old reference with the new donor split.\n\n'
        f'Previous dataset is preserved at `{backup}`. No old data, training runs, or evolution '
        'outputs were deleted.\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--active',type=Path,required=True)
    p.add_argument('--condition',choices=['no_background','with_background'],required=True)
    p.add_argument('--backup-tag',required=True)
    a=p.parse_args();print(json.dumps(activate(a.dataset,a.active,a.condition,a.backup_tag),indent=2))
