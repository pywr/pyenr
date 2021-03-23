import numpy as np

from six import with_metaclass
from pywr.parameters import load_parameter, pop_kwarg_parameter, load_parameter_values
from pywr._core import AbstractNode, Node as BaseNode, Storage as BaseStorage,  StorageInput, StorageOutput, AbstractStorage
from pywr.nodes import Node, NodeMeta, Drawable, Connectable


class Bus(with_metaclass(NodeMeta, Drawable, Connectable, AbstractNode)):
    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        cost = data.pop('cost', 0.0)
        min_flow = data.pop('min_flow', None)
        max_flow = data.pop('max_flow', None)

        data.pop('type')
        node = cls(model=model, name=name, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0
        node.cost = cost
        node.min_flow = min_flow
        node.max_flow = max_flow

        return node


class Generator(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        cost = data.pop('cost', 0.0)
        min_flow = data.pop('min_flow', None)
        max_flow = data.pop('max_flow', None)

        data.pop('type')
        node = cls(model=model, name=name, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0
        node.cost = cost
        node.min_flow = min_flow
        node.max_flow = max_flow

        return node


class PiecewiseGenerator(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    def __init__(self, model, name, **kwargs):
        self.allow_isolated = True
        costs = kwargs.pop('cost')
        max_flows = kwargs.pop('max_flow')

        if len(costs) != len(max_flows):
            raise ValueError("Piecewise max_flow and cost keywords must be the same length.")

        # Setup internall generators
        self.subgenerators = []
        for i, (max_flow, cost) in enumerate(zip(max_flows, costs)):
            generator = Generator(model, name=f'{name} Sub-generator[{i}]')
            generator.max_flow = max_flow
            generator.cost = cost
            self.subgenerators.append(generator)

        super().__init__(model, name, **kwargs)

    def iter_slots(self, slot_name=None, is_connector=True):
        for generator in self.subgenerators:
            yield generator

    def after(self, timestep):
        """
        Set total flow on this link as sum of sublinks
        """
        for generator in self.subgenerators:
            self.commit_all(generator.flow)
        # Make sure save is done after setting aggregated flow
        super().after(timestep)

    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        costs = data.pop('cost')
        max_flows = data.pop('max_flow')
        data.pop('type')
        costs = [load_parameter(model, c) for c in costs]
        max_flows = [load_parameter(model, mf) for mf in max_flows]
        return cls(model, name, cost=costs, max_flow=max_flows, **data)


class Load(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        cost = data.pop('cost', 0.0)
        min_flow = data.pop('min_flow', None)
        max_flow = data.pop('max_flow', None)

        data.pop('type')
        node = cls(model=model, name=name, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0
        node.cost = cost
        node.min_flow = min_flow
        node.max_flow = max_flow

        return node


class Line(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):

    def __init__(self, *args, **kwargs):
        self.reactance = kwargs.pop('reactance', 0.1)
        self.loss = kwargs.pop('loss', 0.0)
        super().__init__(*args, **kwargs)


    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        data.pop('type')
        max_flow = data.pop('max_flow', None)
        node = cls(model=model, name=name, **data)

        max_flow = load_parameter(model, max_flow)
        if max_flow is not None:
            node.max_flow = max_flow

        return node


class Battery(with_metaclass(NodeMeta, Drawable, Connectable, BaseStorage)):
    """A generic storage Node

    In terms of connections in the network the Storage node behaves like any
    other node, provided there is only 1 input and 1 output. If there are
    multiple sub-nodes the connections need to be explicit about which they
    are connecting to. For example:

    >>> storage(model, 'reservoir', num_outputs=1, num_inputs=2)
    >>> supply.connect(storage)
    >>> storage.connect(demand1, from_slot=0)
    >>> storage.connect(demand2, from_slot=1)

    The attribtues of the sub-nodes can be modified directly (and
    independently). For example:

    >>> storage.outputs[0].max_flow = 15.0

    If a recorder is set on the storage node, instead of recording flow it
    records changes in storage. Any recorders set on the output or input
    sub-nodes record flow as normal.
    """
    def __init__(self, model, name, num_outputs=1, num_inputs=1, *args, **kwargs):
        # cast number of inputs/outputs to integer
        # this is needed if values come in as strings sometimes
        num_outputs = int(num_outputs)
        num_inputs = int(num_inputs)

        min_volume = pop_kwarg_parameter(kwargs, 'min_volume', 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, 'max_volume', 0.0)
        initial_volume = kwargs.pop('initial_volume', 0.0)
        initial_volume_pc = kwargs.pop('initial_volume_pc', None)
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)

        position = kwargs.pop("position", {})

        super().__init__(model, name, **kwargs)

        # TODO this doesn't need multiple inputs and outputs
        self.outputs = []
        for n in range(0, num_outputs):
            self.outputs.append(StorageOutput(model, name="[output{}]".format(n), parent=self))

        self.inputs = []
        for n in range(0, num_inputs):
            self.inputs.append(StorageInput(model, name="[input{}]".format(n), parent=self))

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.initial_volume_pc = initial_volume_pc
        self.cost = cost
        self.position = position

        # TODO FIXME!
        # StorageOutput and StorageInput are Cython classes, which do not have
        # NodeMeta as their metaclass, therefore they don't get added to the
        # model graph automatically.
        for node in self.outputs:
            self.model.graph.add_node(node)
        for node in self.inputs:
            self.model.graph.add_node(node)

    def after(self, ts):
        AbstractStorage.after(self, ts)

        for i, si in enumerate(self.model.scenarios.combinations):
            self._volume[i] += self.flow[i]
            # Ensure variable maximum volume is taken in to account

            mxv = self.get_max_volume(si)
            mnv = self.get_min_volume(si)

            if abs(self._volume[i] - mxv) < 1e-6:
                self._volume[i] = mxv
            if abs(self._volume[i] - mnv) < 1e-6:
                self._volume[i] = mnv

            try:
                self._current_pc[i] = self._volume[i] / mxv
            except ZeroDivisionError:
                self._current_pc[i] = np.nan

    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        num_inputs = int(data.pop('inputs', 1))
        num_outputs = int(data.pop('outputs', 1))

        if 'initial_volume' not in data and 'initial_volume_pc' not in data:
            raise ValueError('Initial volume must be specified in absolute or relative terms.')

        initial_volume = data.pop('initial_volume', 0.0)
        initial_volume_pc = data.pop('initial_volume_pc', None)
        max_volume = data.pop('max_volume')
        min_volume = data.pop('min_volume', 0.0)
        cost = data.pop('cost', 0.0)

        data.pop('type', None)
        # Create the instance
        node = cls(model=model, name=name, num_inputs=num_inputs, num_outputs=num_outputs, **data)

        # Load the parameters after the instance has been created to prevent circular
        # loading errors

        # Try to coerce initial volume to float.
        try:
            initial_volume = float(initial_volume)
        except TypeError:
            initial_volume = load_parameter_values(model, initial_volume)
        node.initial_volume = initial_volume
        node.initial_volume_pc = initial_volume_pc

        max_volume = load_parameter(model, max_volume)
        if max_volume is not None:
            node.max_volume = max_volume

        min_volume = load_parameter(model, min_volume)
        if min_volume is not None:
            node.min_volume = min_volume

        cost = load_parameter(model, cost)
        if cost is None:
            cost = 0.0
        node.cost = cost

        return node
