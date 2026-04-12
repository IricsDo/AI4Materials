"""
Bonito Download
"""

import os
import re
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import requests
from tqdm import tqdm


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "dataset/")
__models__ = os.path.join(__dir__, "pretrain/")

class File:
    """
    Small class for downloading models and training assets.
    """
    __url__ = "https://china.scidb.cn/download?fileId=c2487ff51a6230ba2ff55f0b5a8bfe8f&traceId=5bdb3fba-ef44-4f16-a704-645ed4093fb8"
    sb_url = "https://www.superband.work/download/"

    def __init__(self, path, url_frag, force=False):
        self.path = path
        self.force = force
        self.filename = url_frag
        if url_frag.endswith('.hdf5'):
            self.url = self.__url__
            self.fname = self.filename
        else:
            self.url = os.path.join(self.sb_url, url_frag)
            self.fname = self.filename

    def location(self, filename):
        return os.path.join(self.path, filename)

    def exists(self, filename):
        return os.path.exists(self.location(filename))

    def download(self):
        """
        Download the remote file
        """
        # create the requests for the file
        req = requests.get(self.url, stream=True, verify=False)
        total = int(req.headers.get('content-length', 0))
        fname = self.fname

        # skip download if local file is found
        if self.exists(fname.strip('.pt')) and not self.force:
            print("[skipping %s]" % fname, file=sys.stderr)
            return

        # download the file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.location(fname), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded %s]" % fname, file=sys.stderr)

models = [
    "0724.pt"
]

training = [
    "0724.hdf5",
]

def main(args):
    """
    Download models and training sets
    """
    if args.models or args.all:
        if args.show:
            print("[available models]", file=sys.stderr)
            for model in models:
                print(f" - {model}", file=sys.stderr)
        else:
            print("[downloading models]", file=sys.stderr)
            for model in models:
                File(__models__, model, args.force).download()
    if args.training or args.all:
        print("[downloading training data]", file=sys.stderr)
        for train in training:
            File(__data__, train, args.force).download()


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true')
    group.add_argument('--models', action='store_true')
    group.add_argument('--training', action='store_true')
    parser.add_argument('--list', '--show', dest='show', action='store_true')
    parser.add_argument('-f', '--force', action='store_true')
    return parser
