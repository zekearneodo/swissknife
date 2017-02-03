from bci_pipeline import kilosort
import logging
import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser(description='Run bci_pipeline on a computer (niao for now)')
    parser.add_argument('bird', default = '', nargs='?',
                       help='bird that has ss data in the cube')
    parser.add_argument('sess', default = '', nargs='?',
                       help='session')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("do_kilosort")
    logger.info('Will do kilosort on bird {}, sess {}'.format(args.bird, args.sess))
    try:
        kilosort.run_kilosort(args.bird, args.sess, no_copy=False)
        logger.info('Finished sorting')
    except:
        logger.error('Something went wrong')
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
