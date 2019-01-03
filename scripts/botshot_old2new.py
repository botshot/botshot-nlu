#!/usr/bin/env python

import argparse
import os, yaml


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("source", type=str)
    args = argp.parse_args()
    source = args.source
    with open(source) as fp:
        data = yaml.load(fp)
    out_lines = []
    for obj in data['data']:
        value = obj['value']
        examples = obj['samples']
        out_lines.append('@' + value)
        for example in examples:
            out_lines.append(example)
        out_lines.append(" ")
    
    with open("output.txt", "w") as fp:
        for line in out_lines:
            fp.write(line + "\n")
    print("Done.")


if __name__ == "__main__":
    main()

