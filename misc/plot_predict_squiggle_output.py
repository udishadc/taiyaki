#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt

from taiyaki import fileio

if True:
    #  Protect in block to prevent autopep8 refactoring
    import matplotlib
    matplotlib.use('Agg')


def main():
    print('Plots output of predict_squiggle.py\nUsage:\n' +
          'plot_predict_squiggle_output.py <predict_squiggle_output_file> ' +
          '<output_png_file>')
    if len(sys.argv) < 3:
        print("ERROR: Needs command line arguments!")
    else:
        predict_squiggle_output_file = sys.argv[1]
        plotfile = sys.argv[2]
        t = fileio.readtsv(predict_squiggle_output_file)

        plt.figure(figsize=(16, 5))
        tstart = 0
        for nrow in range(len(t)):
            i, sd, dwell = t['current'][nrow], t['sd'][nrow], t['dwell'][nrow]
            centret = tstart + dwell / 2
            plt.bar(centret, sd, dwell, i - sd / 2)
            plt.text(centret, i, t['base'][nrow])
            tstart += dwell
        plt.xlabel('time')
        plt.ylabel('current')
        plt.grid()
        plt.savefig(plotfile)


if __name__ == '__main__':
    main()
