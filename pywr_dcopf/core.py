from six import with_metaclass
from pywr.parameters import load_parameter, pop_kwarg_parameter, load_parameter_values
from pywr._core import AbstractNode, Node as BaseNode, Storage as BaseStorage,  StorageInput, StorageOutput
from pywr.nodes import Node, NodeMeta, Drawable, Connectable
from pywr.schema import NodeSchema, fields
import marshmallow


class Bus(with_metaclass(NodeMeta, Drawable, Connectable, AbstractNode)):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_flow = fields.ParameterReferenceField(allow_none=True)
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)


class Generator(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_flow = fields.ParameterReferenceField(allow_none=True)
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)


class PiecewiseGenerator(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_flows = marshmallow.fields.List(fields.ParameterField(allow_none=True))
        costs = marshmallow.fields.List(fields.ParameterField(allow_none=True))

    def __init__(self, model, name, **kwargs):
        self.allow_isolated = True
        costs = kwargs.pop('costs')
        max_flows = kwargs.pop('max_flows')

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


class Load(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_flow = fields.ParameterReferenceField(allow_none=True)
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)


class Line(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_flow = fields.ParameterReferenceField(allow_none=True)
        reactance = marshmallow.fields.Number()
        cost = fields.ParameterReferenceField(allow_none=True)

    def __init__(self, *args, **kwargs):
        self.reactance = kwargs.pop('reactance', 0.1)
        super().__init__(*args, **kwargs)


class Battery(with_metaclass(NodeMeta, Drawable, Connectable, BaseStorage)):
    """A generic storage Node

    In terms of connections in the network the Storage node behaves like any
    other node, provided there is only 1 input and 1 output. If there are
    multiple sub-nodes the connections need to be explicit about which they
    are connecting to. For example:

    >>> storage(model, 'reservoir', outputs=1, inputs=2)
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
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_volume = fields.ParameterReferenceField(required=False)
        min_volume = fields.ParameterReferenceField(required=False)
        cost = fields.ParameterReferenceField(required=False)
        initial_volume = fields.ParameterValuesField(required=False)
        initial_volume_pc = marshmallow.fields.Number(required=False)
        inputs = marshmallow.fields.Integer(required=False, default=1)
        outputs = marshmallow.fields.Integer(required=False, default=1)

    def __init__(self, model, name, outputs=1, inputs=1, *args, **kwargs):
        # cast number of inputs/outputs to integer
        # this is needed if values come in as strings sometimes
        outputs = int(outputs)
        inputs = int(inputs)

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
        for n in range(0, outputs):
            self.outputs.append(StorageOutput(model, name="[output{}]".format(n), parent=self))

        self.inputs = []
        for n in range(0, inputs):
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
