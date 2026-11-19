from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info('Central logging initialized successfully.')


if __name__ == '__main__':
    main()
