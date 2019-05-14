from six import with_metaclass
from pywr.parameters import load_parameter
from pywr._core import AbstractNode, Node as BaseNode
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
        super().__init__(*args, **kwargs)


    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        data.pop('type')
        node = cls(model=model, name=name, **data)
        return node
