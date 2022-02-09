"""Run script for PV model examples.
"""
from pywr.model import Model
from pywr.recorders import NumpyArrayNodeRecorder, CSVRecorder
from pywr.recorders.progress import ProgressRecorder
from pywr_dcopf import core
import pandas
from matplotlib import pyplot as plt
import os
import sys
import logging
logger = logging.getLogger(__name__)


def main(filename):
    base, ext = os.path.splitext(filename)
    m = Model.load(filename, solver='glpk-dcopf')

    gen1 = NumpyArrayNodeRecorder(m, m.nodes['gen1'])
    pv2 = NumpyArrayNodeRecorder(m, m.nodes['pv2'])
    ProgressRecorder(m)
    CSVRecorder(m, f'{base}.csv')

    m.setup()
    stats = m.run()
    print(stats.to_dataframe())

    df = pandas.concat({'gen1': gen1.to_dataframe(), 'pv2': pv2.to_dataframe()}, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(ax=ax)
    df.resample('D').mean().plot(ax=ax, color='black')
    ax.set_ylabel('MW')
    fig.savefig(f'{base}.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    df.resample('M').sum().plot(ax=ax)
    ax.set_ylabel('MWh per month')
    fig.savefig(f'{base}-monthly.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    import sys
    pywr_logger = logging.getLogger('pywr')
    pywr_logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)
    pywr_logger.addHandler(ch)

    main(sys.argv[1])

