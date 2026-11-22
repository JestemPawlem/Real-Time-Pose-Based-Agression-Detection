from pathlib import Path

from src.utils import get_logger
from src.utils.video import VIDEO_EXTENSIONS, get_video_metadata

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'RWF-2000'
CORRUPTED_REPORT_PATH = PROJECT_ROOT / 'logs' / 'corrupted_videos.txt'


def verify_dataset(dataset_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """Scan dataset directories dynamically and verify video file integrity."""
    if not dataset_dir.exists():
        logger.error(f'Dataset directory not found: {dataset_dir}')
        return False

    split_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not split_dirs:
        logger.error(f'No split subdirectories found in: {dataset_dir}')
        return False

    total_checked = 0
    total_valid = 0
    corrupted_files: list[Path] = []

    for split_dir in split_dirs:
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for class_dir in class_dirs:
            videos = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
            valid_in_class = 0

            for video_path in videos:
                total_checked += 1
                if get_video_metadata(video_path) is not None:
                    valid_in_class += 1
                    total_valid += 1
                else:
                    corrupted_files.append(video_path)
                    logger.warning(f'Corrupted or unreadable video: {video_path}')

            logger.info(f'{split_dir.name}/{class_dir.name}: {valid_in_class}/{len(videos)} valid ')

    logger.info(
        f'Verification finished: {total_valid}/{total_checked} valid clips, {len(corrupted_files)} corrupted.'
    )

    if corrupted_files:
        CORRUPTED_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CORRUPTED_REPORT_PATH.write_text(
            '\n'.join(str(p.relative_to(PROJECT_ROOT)) for p in corrupted_files) + '\n',
            encoding='utf-8',
        )
        logger.warning(f'Saved list of corrupted files to: {CORRUPTED_REPORT_PATH}')
        return False

    if CORRUPTED_REPORT_PATH.exists():
        CORRUPTED_REPORT_PATH.unlink()

    logger.info('All discovered video files passed integrity checks.')
    return True


if __name__ == '__main__':
    success = verify_dataset()
    raise SystemExit(0 if success else 1)
