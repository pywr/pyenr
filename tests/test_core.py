from pywr.model import Model
from pywr.recorders import NumpyArrayNodeRecorder
from pywr_dcopf.core import Bus, Line, Generator, Load
import numpy as np
import pandas
import os


TEST_FOLDER = os.path.dirname(__file__)


def test_core():
    """Test basic functionality of the DC-OPF model."""
    m = Model(solver='glpk-dcopf')

    g1 = Generator(m, 'gen1')
    g1.max_flow = 100
    g1.cost = 1.0

    g2 = Generator(m, 'gen2')
    g2.max_flow = 100
    g2.cost = 2.0

    l3 = Load(m, 'load3')
    l3.max_flow = 150
    l3.cost = -10

    b1 = Bus(m, 'bus1')
    b2 = Bus(m, 'bus2')
    b3 = Bus(m, 'bus3')

    l12 = Line(m, 'line12')
    l13 = Line(m, 'line13')
    l23 = Line(m, 'line23')

    g1.connect(b1)
    g2.connect(b2)
    l3.connect(b3)

    b1.connect(l12)
    l12.connect(b2)

    b1.connect(l13)
    l13.connect(b3)

    b2.connect(l23)
    l23.connect(b3)

    m.setup()
    m.run()

    np.testing.assert_allclose(g1.flow, [100.0])
    np.testing.assert_allclose(g2.flow, [50.0])
    np.testing.assert_allclose(l3.flow, [150.0])
    # TODO test the flow in each line (not currently updated by the solver)


def test_ghana_load():

    m = Model.load(os.path.join(TEST_FOLDER, 'models', 'ghana-pywr.json'), solver='glpk-dcopf')

    m.setup()
    m.run()

    # TODO add assertions


def test_simple_pv():

    m = Model.load(os.path.join(TEST_FOLDER, 'models', 'simple-pv.json'), solver='glpk-dcopf')

    gen1 = NumpyArrayNodeRecorder(m, m.nodes['gen1'])
    pv2 = NumpyArrayNodeRecorder(m, m.nodes['pv2'])

    m.setup()
    m.run()

    df = pandas.concat({'gen1': gen1.to_dataframe(), 'pv2': pv2.to_dataframe()}, axis=1)

    assert df.shape[0] == 745
    # TODO add better assertions

