from pywr._core cimport AbstractNode, Node, Timestep
import numpy as np
cimport numpy as np


# cdef class Bus(Node):
#
#     property phase:
#         """Voltage phase in this bus
#         """
#         def __get__(self):
#             return np.array(self._phase)
#
#     cpdef setup(self, model):
#         """Called before the first run of the model"""
#         Node.setup(self, model)
#         cdef int ncomb = len(model.scenarios.combinations)
#         self._phase = np.empty(ncomb, dtype=np.float64)
#
#     cpdef reset(self):
#         """Called at the beginning of a run"""
#         Node.reset(self)
#         cdef int i
#         for i in range(self._phase.shape[0]):
#             self._phase[i] = 0.0
#
#     cpdef before(self, Timestep ts):
#         """Called at the beginning of the timestep"""
#         Node.before(self, ts)
#         cdef int i
#         for i in range(self._flow.shape[0]):
#             self._flow[i] = 0.0
#
#     cpdef commit(self, int scenario_index, double flow, double phase):
#         """Called once for each route the node is a member of"""
#         Node.commit(self, scenario_index, flow)
#         self._phase[scenario_index] += phase
#
#     cpdef commit_all(self, double[:] flow, double[:] phase):
#         """Called once for each route the node is a member of"""
#         Node.commit_all(self, flow)
#         cdef int i
#         for i in range(self._phase.shape[0]):
#             self._phase[i] += phase[i]