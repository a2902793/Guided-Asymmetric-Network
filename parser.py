import argparse, sys
import options.options as option

parser = argparse.ArgumentParser()
parser.add_argument('--file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
opt = parser.parse_args().opt
print(opt)