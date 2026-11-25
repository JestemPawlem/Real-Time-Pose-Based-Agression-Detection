import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, SPLIT_MANIFEST_PATH
from src.utils.logger import get_logger
from src.utils.video import compute_video_fingerprint, dhash_distance_matrix

logger = get_logger(__name__)

SPLITS = ('train', 'val', 'test')


def check_leakage(
    df: pd.DataFrame,
    raw_dir: Path,
    hamming_threshold: int = 4,
    workers: int = 8,
) -> list[str]:
    """Check for exact SHA-256 duplicates and matching camera backgrounds across splits."""
    errors: list[str] = []
    paths = [raw_dir / rel for rel in df['relative_path']]

    logger.info('Computing SHA-256 and background dHash for %d videos...', len(paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(compute_video_fingerprint, paths))

    sha_by_split: dict[str, dict[str, str]] = {s: {} for s in SPLITS}
    dhash_by_split: dict[str, tuple[list[str], list[np.ndarray]]] = {s: ([], []) for s in SPLITS}

    for row, (sha256, dhash) in zip(df.itertuples(index=False), results, strict=True):
        split = str(row.split)
        rel_path = str(row.relative_path)
        sha_by_split[split][sha256] = rel_path
        if dhash is not None:
            dhash_by_split[split][0].append(rel_path)
            dhash_by_split[split][1].append(dhash)

    # 1. Exact binary duplicates across splits
    for split_a, split_b in itertools.combinations(SPLITS, 2):
        shared = sorted(set(sha_by_split[split_a]) & set(sha_by_split[split_b]))
        if shared:
            errors.append(
                f"SHA-256 leakage between '{split_a}' and '{split_b}': {len(shared)} file(s)"
            )
            for digest in shared:
                logger.error(
                    '  Duplicate: %s  <-->  %s',
                    sha_by_split[split_a][digest],
                    sha_by_split[split_b][digest],
                )
        else:
            logger.info("PASS SHA-256: '%s' vs '%s' (0 duplicates)", split_a, split_b)

    # 2. Static camera background similarity across splits
    for split_a, split_b in itertools.combinations(SPLITS, 2):
        paths_a, hashes_a = dhash_by_split[split_a]
        paths_b, hashes_b = dhash_by_split[split_b]
        if not hashes_a or not hashes_b:
            continue

        dists = dhash_distance_matrix(np.vstack(hashes_a), np.vstack(hashes_b))
        idx_a, idx_b = np.where(dists <= hamming_threshold)
        if len(idx_a) > 0:
            errors.append(
                f"Background dHash leakage between '{split_a}' and '{split_b}': "
                f'{len(idx_a)} pair(s) with Hamming <= {hamming_threshold}'
            )
            for i, j in zip(idx_a.tolist(), idx_b.tolist(), strict=True):
                logger.error(
                    '  dist=%d | %s  <-->  %s',
                    int(dists[i, j]),
                    paths_a[i],
                    paths_b[j],
                )
        else:
            logger.info(
                "PASS background dHash: '%s' vs '%s' (0 pairs <= %d)",
                split_a,
                split_b,
                hamming_threshold,
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate split manifest against binary and cross-camera leakage.'
    )
    parser.add_argument('--manifest', type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument('--raw-dir', type=Path, default=RAW_DATA_DIR)
    parser.add_argument('--hamming-threshold', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    errors = check_leakage(
        df=df,
        raw_dir=args.raw_dir,
        hamming_threshold=args.hamming_threshold,
        workers=args.workers,
    )

    if errors:
        logger.error('Validation FAILED (%d error(s))', len(errors))
        sys.exit(1)

    logger.info('Validation PASSED.')


if __name__ == '__main__':
    main()
