"""
  This module defines types used by the glpk-dcopf solver.
"""
import numpy as np

from pywr.parameters import (
    load_parameter,
    pop_kwarg_parameter,
    load_parameter_values
)

from pywr.nodes import (
    NodeMeta,
    Drawable,
    Connectable,
    Loadable
)

from pywr._core import (
    Node as BaseNode,
    Storage as BaseStorage,
    StorageInput,
    StorageOutput,
    AbstractStorage
)

__all__ = (
    "Battery",
    "Bus",
    "Generator",
    "Line",
    "Load",
    "PiecewiseGenerator"
)


class Bus(BaseNode, Loadable, Drawable, Connectable, metaclass=NodeMeta):

    def __init__(self, model, name, *args, **kwargs):
        max_flow = kwargs.pop("max_flow", None)
        min_flow = kwargs.pop("min_flow", None)
        cost = kwargs.pop("cost", None)
        kwargs.pop("position", None)
        super().__init__(model, name, *args, **kwargs)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)

        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0

        self.cost = cost
        self.min_flow = min_flow
        self.max_flow = max_flow


class Generator(BaseNode, Loadable, Connectable, Drawable, metaclass=NodeMeta):

    def __init__(self, model, name, *args, **kwargs):
        max_flow = kwargs.pop("max_flow", None)
        min_flow = kwargs.pop("min_flow", None)
        cost = kwargs.pop("cost", None)
        kwargs.pop("position", None)
        super().__init__(model, name, *args, **kwargs)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)

        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0

        self.cost = cost
        self.min_flow = min_flow
        self.max_flow = max_flow


class PiecewiseGenerator(BaseNode, Loadable, Drawable, Connectable, metaclass=NodeMeta):

    def __init__(self, model, name, *args, **kwargs):
        self.allow_isolated = True
        costs = kwargs.pop('cost')
        max_flows = kwargs.pop('max_flow')
        kwargs.pop("position", None)

        if len(costs) != len(max_flows):
            raise ValueError("Piecewise max_flow and cost keywords must be the same length.")

        # Setup internal generators
        self.subgenerators = []
        for i, (max_flow, cost) in enumerate(zip(max_flows, costs)):
            generator = Generator(model, name=f'{name} Sub-generator[{i}]')
            generator.max_flow = max_flow
            generator.cost = cost
            self.subgenerators.append(generator)

        super().__init__(model, name, *args, **kwargs)

    def iter_slots(self, slot_name=None, is_connector=True):
        for generator in self.subgenerators:
            yield generator

    def after(self, timestep, adjustment=None):
        """  Set total flow on this link as sum of sublinks """
        for generator in self.subgenerators:
            self.commit_all(generator.flow)
        # Make sure save is done after setting aggregated flow
        super().after(timestep)


class Load(BaseNode, Loadable, Connectable, Drawable, metaclass=NodeMeta):

    def __init__(self, model, name, **kwargs):
        min_flow = kwargs.pop('min_flow', None)
        max_flow = kwargs.pop('max_flow', None)
        cost = kwargs.pop('cost', None)
        kwargs.pop("position", None)
        super().__init__(model, name, **kwargs)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)

        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = 0.0

        self.cost = cost
        self.min_flow = min_flow
        self.max_flow = max_flow


class Line(BaseNode, Loadable, Connectable, Drawable, metaclass=NodeMeta):

    default_reactance = 0.1

    def __init__(self, model, name, *args, **kwargs):
        self.reactance = kwargs.pop('reactance', Line.default_reactance)
        self.loss = kwargs.pop('loss', 0.0)
        max_flow = kwargs.pop('max_flow', None)
        kwargs.pop("position", None)
        super().__init__(model, name, *args, **kwargs)

        max_flow = load_parameter(model, max_flow)
        if max_flow is not None:
            self.max_flow = max_flow


class Battery(BaseStorage, Loadable, Connectable, Drawable, metaclass=NodeMeta):
    """ Shares the behaviour of a generic Storage Node.

        In terms of connections in the network the Storage node behaves like any
        other node, provided there is only 1 input and 1 output. If there are
        multiple sub-nodes the connections need to be explicit about which they
        are connecting to. For example:

        >>> storage(model, 'reservoir', num_outputs=1, num_inputs=2)
        >>> supply.connect(storage)
        >>> storage.connect(demand1, from_slot=0)
        >>> storage.connect(demand2, from_slot=1)

        The attributes of the sub-nodes can be modified directly (and
        independently). For example:

        >>> storage.outputs[0].max_flow = 15.0

        If a recorder is set on the storage node, instead of recording flow it
        records changes in storage. Any recorders set on the output or input
        sub-nodes record flow as normal.
    """
    def __init__(self, model, name, num_outputs=1, num_inputs=1, *args, **kwargs):
        #  Ensure num_inputs/num_outputs are ints
        try:
            num_outputs = int(num_outputs)
            num_inputs = int(num_inputs)
        except (TypeError, ValueError) as err:
            raise err.__class__(f"Invalid argument for num_inputs/num_outputs: {str(err)}")

        if "initial_volume" not in kwargs and "initial_volume_pc" not in kwargs:
            raise ValueError("Initial volume must be specified in absolute or relative terms.")

        min_volume = pop_kwarg_parameter(kwargs, 'min_volume', 0.0)
        max_volume = pop_kwarg_parameter(kwargs, 'max_volume', 0.0)

        initial_volume = kwargs.pop('initial_volume', 0.0)
        try:
            initial_volume = float(initial_volume)
        except (TypeError, ValueError):
            initial_volume = load_parameter_values(model, initial_volume)

        initial_volume_pc = kwargs.pop('initial_volume_pc', None)
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)

        position = kwargs.pop("position", {})

        super().__init__(model, name, *args, **kwargs)

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

    def after(self, ts, adjustment=None):
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
