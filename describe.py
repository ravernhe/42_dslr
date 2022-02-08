import argparse
from srcs.describe_class import *

def main(filename):
    description = Describe(filename)
    description.explain()
    description.show()
    return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help="Input dataset")
  args = parser.parse_args()

  main(args.dataset)