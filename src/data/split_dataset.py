import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_SEED, RAW_DATA_DIR, SPLIT_MANIFEST_PATH
from src.utils.logger import get_logger
from src.utils.video import (
    VIDEO_EXTENSIONS,
    compute_video_fingerprint,
    dhash_distance_matrix,
    extract_source_group,
    get_video_metadata,
)

logger = get_logger(__name__)


def collect_dataset_records(raw_dir: Path) -> pd.DataFrame:
    """Scan raw data directory and return metadata for all valid clips."""
    records: list[dict] = []

    for split_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
            for video in videos:
                metadata = get_video_metadata(video)
                if metadata is None:
                    logger.warning('Skipping unreadable video: %s', video)
                    continue

                records.append(
                    {
                        'original_split': split_dir.name,
                        'label': class_dir.name,
                        'relative_path': video.relative_to(raw_dir).as_posix(),
                        'source_group': extract_source_group(video),
                        'width': metadata.width,
                        'height': metadata.height,
                        'fps': metadata.fps,
                        'frame_count': metadata.frame_count,
                        'duration_sec': metadata.duration_sec,
                    }
                )

    return pd.DataFrame(records)


def assign_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Keep original train and split original val into val and test."""
    result = df.copy()
    result['split'] = ''

    train_mask = result['original_split'] == 'train'
    val_mask = result['original_split'] == 'val'

    result.loc[train_mask, 'split'] = 'train'

    val_df = result.loc[val_mask]
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_rel_idx, test_rel_idx = next(
        sgkf.split(val_df, y=val_df['label'], groups=val_df['source_group'])
    )

    result.loc[val_df.index[val_rel_idx], 'split'] = 'val'
    result.loc[val_df.index[test_rel_idx], 'split'] = 'test'

    ordered_cols = [
        'split',
        'label',
        'relative_path',
        'source_group',
        'width',
        'height',
        'fps',
        'frame_count',
        'duration_sec',
    ]
    return result[ordered_cols].sort_values(['split', 'relative_path'], ignore_index=True)


def log_split_summary(df: pd.DataFrame) -> None:
    """Log split sizes and class balance."""
    total = len(df)
    logger.info('Total clips in manifest: %s', total)

    for split_name in ('train', 'val', 'test'):
        subset = df[df['split'] == split_name]
        counts = subset['label'].value_counts().to_dict()
        logger.info(
            'Split %-5s | %4d clips (%5.1f%%) | classes=%s',
            split_name,
            len(subset),
            len(subset) / total * 100,
            counts,
        )


def deduplicate_records(
    df: pd.DataFrame,
    raw_dir: Path,
    hamming_threshold: int = 4,
    workers: int = 8,
) -> pd.DataFrame:
    """Remove exact SHA-256 duplicates and val clips matching train backgrounds."""
    work = df.sort_values(['original_split', 'relative_path'], ignore_index=True).copy()
    paths = [raw_dir / rel for rel in work['relative_path']]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(compute_video_fingerprint, paths))

    work['sha256'] = [sha for sha, _ in results]
    dhashes = [dhash for _, dhash in results]

    # 1. Exact SHA-256 deduplication across the entire dataset
    dup_mask = work['sha256'].duplicated(keep='first')
    if dup_mask.any():
        logger.info('Dropping %d exact SHA-256 duplicate(s):', int(dup_mask.sum()))
        for rel in work.loc[dup_mask, 'relative_path']:
            logger.info('  - %s', rel)
        keep_indices = work.index[~dup_mask].tolist()
        work = work.loc[keep_indices].reset_index(drop=True)
        dhashes = [dhashes[i] for i in keep_indices]

    # 2. Drop val clips visually matching any train background (dHash <= threshold)
    train_idx = [
        i
        for i, split in enumerate(work['original_split'])
        if split == 'train' and dhashes[i] is not None
    ]
    val_idx = [
        i
        for i, split in enumerate(work['original_split'])
        if split == 'val' and dhashes[i] is not None
    ]

    drop_val_indices: set[int] = set()
    if train_idx and val_idx:
        train_arr = np.vstack([dhashes[i] for i in train_idx])
        val_arr = np.vstack([dhashes[i] for i in val_idx])
        dists = dhash_distance_matrix(val_arr, train_arr)

        for local_v, global_v in enumerate(val_idx):
            min_t = int(np.argmin(dists[local_v]))
            min_dist = int(dists[local_v, min_t])
            if min_dist <= hamming_threshold:
                drop_val_indices.add(global_v)
                logger.info(
                    'Dropping near-duplicate val clip (dHash dist=%d): %s (matches %s)',
                    min_dist,
                    work.at[global_v, 'relative_path'],
                    work.at[train_idx[min_t], 'relative_path'],
                )

    if drop_val_indices:
        keep_indices = [i for i in range(len(work)) if i not in drop_val_indices]
        work = work.loc[keep_indices].reset_index(drop=True)
        dhashes = [dhashes[i] for i in keep_indices]

    # 3. Merge source_group for any visually matching clips inside val before val/test split
    rem_val_idx = [
        i
        for i, split in enumerate(work['original_split'])
        if split == 'val' and dhashes[i] is not None
    ]
    if len(rem_val_idx) > 1:
        val_arr = np.vstack([dhashes[i] for i in rem_val_idx])
        dists = dhash_distance_matrix(val_arr, val_arr)
        i_idx, j_idx = np.where(
            (dists <= hamming_threshold) & (np.triu(np.ones_like(dists), 1) > 0)
        )
        for a, b in zip(i_idx.tolist(), j_idx.tolist(), strict=True):
            ga = work.at[rem_val_idx[a], 'source_group']
            gb = work.at[rem_val_idx[b], 'source_group']
            if ga != gb:
                work.loc[work['source_group'] == gb, 'source_group'] = ga

    return work.drop(columns=['sha256'])


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate reproducible 80/10/10 train/val/test split manifest for RWF-2000.'
    )
    parser.add_argument('--raw-dir', type=Path, default=RAW_DATA_DIR)
    parser.add_argument('--output', type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    logger.info('Scanning raw dataset in: %s', args.raw_dir)
    df = collect_dataset_records(args.raw_dir)
    df = deduplicate_records(df, args.raw_dir)
    split_df = assign_splits(df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, index=False)
    logger.info('Saved split manifest to: %s', args.output)
    log_split_summary(split_df)


if __name__ == '__main__':
    main()
