from pywr.model import Model
from pywr.recorders import NumpyArrayNodeRecorder
from pywr.recorders.progress import ProgressRecorder
from pywr_dcopf import core
import pandas
from matplotlib import pyplot as plt
import logging
logger = logging.getLogger(__name__)


def main():
    m = Model.load('simple-pv.json', solver='glpk-dcopf')

    gen1 = NumpyArrayNodeRecorder(m, m.nodes['gen1'])
    pv2 = NumpyArrayNodeRecorder(m, m.nodes['pv2'])
    ProgressRecorder(m)

    m.setup()
    stats = m.run()
    print(stats.to_dataframe())

    df = pandas.concat({'gen1': gen1.to_dataframe(), 'pv2': pv2.to_dataframe()}, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(ax=ax)
    ax.set_ylabel('MW')
    fig.savefig('simple-pv-hourly.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 4))
    df.resample('M').sum().plot(ax=ax)
    ax.set_ylabel('MWh per month')
    fig.savefig('simple-pv-monthly.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    pywr_logger = logging.getLogger('pywr')
    pywr_logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)
    pywr_logger.addHandler(ch)

    main()
