from libc.stdlib cimport malloc, free
from libc.math cimport abs
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

cimport cython
from collections import defaultdict
from pywr_dcopf.core import Bus, Line, Generator, Load, Battery
from pywr._core cimport *
from pywr.core import ModelStructureError
import time

include "glpk.pxi"

inf = float('inf')


cdef class CythonGLPKDCOPFSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp

    cdef int idx_col_generation
    cdef int idx_col_load
    cdef int idx_col_phase
    cdef int idx_col_line_losses
    cdef int idx_row_power_flow
    cdef int idx_row_line_capacity
    cdef int idx_row_line_losses
    cdef int idx_row_batteries

    cdef public list buses
    cdef public list lines
    cdef public list generators
    cdef public list loads
    cdef public list batteries

    cdef list all_nodes
    cdef int num_nodes
    cdef int num_routes
    cdef int num_storages
    cdef int num_scenarios
    cdef public object stats

    # Internal representation of the basis for each scenario
    cdef int[:, :] row_stat
    cdef int[:, :] col_stat
    cdef bint is_first_solve
    cdef bint has_presolved
    cdef public bint use_presolve
    cdef public bint save_routes_flows
    cdef public bint retry_solve

    def __cinit__(self):
        # create a new problem
        self.prob = glp_create_prob()

    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level='error',
                 save_routes_flows=False, retry_solve=False):
        self.stats = None
        self.is_first_solve = True
        self.has_presolved = False
        self.use_presolve = use_presolve
        self.save_routes_flows = save_routes_flows
        self.retry_solve = retry_solve

        # Set solver options
        glp_init_smcp(&self.smcp)
        self.smcp.msg_lev = message_levels[message_level]
        if time_limit is not None:
            self.smcp.tm_lim = time_limit
        if iteration_limit is not None:
            self.smcp.it_lim = iteration_limit

        glp_term_hook(term_hook, NULL)

    def __dealloc__(self):
        # free the problem
        glp_delete_prob(self.prob)

    def setup(self, model):
        cdef int* ind
        cdef double* val

        self.all_nodes = list(sorted(model.graph.nodes(), key=lambda n: n.fully_qualified_name))

        graph = model.graph.to_undirected()

        buses = []
        lines = []
        generators = []
        loads = []
        batteries = []

        for some_node in self.all_nodes:
            if isinstance(some_node, Bus):
                buses.append(some_node)
            elif isinstance(some_node, Line):
                lines.append(some_node)
            elif isinstance(some_node, Generator):
                generators.append(some_node)
            elif isinstance(some_node, Load):
                loads.append(some_node)
            elif isinstance(some_node, Battery):
                batteries.append(some_node)
                for generator in some_node.inputs:
                    generators.append(generator)
                for load in some_node.outputs:
                    loads.append(load)

        # clear the previous problem
        glp_erase_prob(self.prob)
        glp_set_obj_dir(self.prob, GLP_MIN)
        # add a column for each route
        self.idx_col_generation = glp_add_cols(self.prob, <int>(len(generators)))
        self.idx_col_load = glp_add_cols(self.prob, <int>(len(loads)))
        self.idx_col_phase = glp_add_cols(self.prob, <int>(len(buses)))
        # Two loss columns (+/-) for each line
        self.idx_col_line_losses = glp_add_cols(self.prob, <int>(2*len(lines)))

        # explicitly set bounds on route and demand columns
        for col, route in enumerate(generators):
            set_col_bnds(self.prob, self.idx_col_generation+col, GLP_LO, 0.0, DBL_MAX)
        for col, route in enumerate(loads):
            set_col_bnds(self.prob, self.idx_col_load+col, GLP_LO, 0.0, DBL_MAX)
        for col, route in enumerate(buses):
            # TODO what should the bounds constraints on the buses be??
            set_col_bnds(self.prob, self.idx_col_phase+col, GLP_LO, 0.0, DBL_MAX)
        for col in range(2*len(lines)):
            set_col_bnds(self.prob, self.idx_col_line_losses+col, GLP_LO, 0.0, DBL_MAX)

        # Power flow constraints
        self.idx_row_power_flow = glp_add_rows(self.prob, len(buses))
        self.idx_row_line_losses = glp_add_rows(self.prob, 2*len(lines))
        iloss = 0
        for ibus, bus in enumerate(buses):
            cols = defaultdict(lambda: 0.0)

            for some_node in graph.neighbors(bus):
                if isinstance(some_node, Line):
                    # (bus) <-> (some_node) <-> (other_bus)

                    # Add line entries
                    susceptance = 1 / some_node.reactance
                    other_bus = [b for b in graph.neighbors(some_node) if b is not bus]
                    assert len(other_bus) == 1
                    other_bus = other_bus[0]
                    assert isinstance(other_bus, Bus)
                    other_ibus = buses.index(other_bus)

                    cols[self.idx_col_phase + ibus] += -susceptance
                    cols[self.idx_col_phase + other_ibus] += susceptance
                    # Add loss to bus power balance
                    cols[self.idx_col_line_losses + iloss] += -1.0
                    # print(ibus, bus, some_node, iloss)
                    # Create a constraint for loss coming into bus from other_bus
                    ind = <int*>malloc(4 * sizeof(int))
                    val = <double*>malloc(4 * sizeof(double))
                    # Loss > line flow
                    ind[1] = self.idx_col_line_losses + iloss
                    val[1] = -1.0
                    ind[2] = self.idx_col_phase + ibus
                    val[2] = -susceptance * some_node.loss
                    ind[3] = self.idx_col_phase + other_ibus
                    val[3] = susceptance * some_node.loss

                    set_mat_row(self.prob, self.idx_row_line_losses+iloss, 3, ind, val)
                    set_row_bnds(self.prob, self.idx_row_line_losses+iloss, GLP_UP, 0.0, 0.0)
                    free(ind)
                    free(val)
                    iloss += 1

                elif isinstance(some_node, Generator):
                    igen = generators.index(some_node)
                    cols[self.idx_col_generation+igen] += 1.0
                elif isinstance(some_node, Load):
                    iload = loads.index(some_node)
                    cols[self.idx_col_load+iload] += -1.0
                elif isinstance(some_node, Battery):
                    # Add the battery's generators and loads to the bus connectivity.
                    for generator in some_node.inputs:
                        igen = generators.index(generator)
                        cols[self.idx_col_generation+igen] += 1.0
                    for load in some_node.outputs:
                        iload = loads.index(load)
                        cols[self.idx_col_load+iload] += -1.0

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, (c, v) in enumerate(cols.items()):
                ind[1+n] = c
                val[1+n] = v
            set_mat_row(self.prob, self.idx_row_power_flow+ibus, len(cols), ind, val)
            set_row_bnds(self.prob, self.idx_row_power_flow+ibus, GLP_FX, 0.0, 0.0)
            free(ind)
            free(val)

        # Line constraints
        self.idx_row_line_capacity = glp_add_rows(self.prob, len(lines))
        for iline, line in enumerate(lines):

            cols = []
            susceptance = 1 / line.reactance

            for bus in graph.neighbors(line):
                ibus = buses.index(bus)
                cols.append(self.idx_col_phase + ibus)

            assert len(cols) == 2

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))

            ind[1] = cols[0]
            val[1] = susceptance
            ind[2] = cols[1]
            val[2] = -susceptance

            set_mat_row(self.prob, self.idx_row_line_capacity+iline, len(cols), ind, val)
            set_row_bnds(self.prob, self.idx_row_line_capacity+iline, GLP_FR, inf_to_dbl_max(-inf), inf_to_dbl_max(inf))
            free(ind)
            free(val)

        # Battery capacity constraints
        if len(batteries):
            self.idx_row_batteries = glp_add_rows(self.prob, len(batteries))
        for row, battery in enumerate(batteries):

            cols_output = []
            for load in battery.outputs:
                cols_output.append(loads.index(load))
            cols_input = []
            for generator in battery.inputs:
                cols_input.append(generators.index(generator))

            ind = <int*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(int))
            val = <double*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(double))
            for n, c in enumerate(cols_output):
                ind[1+n] = self.idx_col_load+c
                val[1+n] = 1
            for n, c in enumerate(cols_input):
                ind[1+len(cols_output)+n] = self.idx_col_generation+c
                val[1+len(cols_output)+n] = -1

            set_mat_row(self.prob, self.idx_row_batteries+row, len(cols_output)+len(cols_input), ind, val)
            free(ind)
            free(val)

        self.buses = buses
        self.lines = lines
        self.generators = generators
        self.loads = loads
        self.batteries = batteries

        self._init_basis_arrays(model)
        self.is_first_solve = True
        self.has_presolved = False

        # reset stats
        self.stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'objective_update': 0.0,
            'bounds_update_power_flow': 0.0,
            'number_of_rows': glp_get_num_rows(self.prob),
            'number_of_cols': glp_get_num_cols(self.prob),
            'number_of_nonzero': glp_get_num_nz(self.prob),
            'number_of_buses': len(buses),
            'number_of_lines': len(lines),
            'number_of_nodes': len(self.all_nodes)
        }

    cdef _init_basis_arrays(self, model):
        """ Initialise the arrays used for storing the LP basis by scenario """
        cdef int nscenarios = len(model.scenarios.combinations)
        cdef int nrows = glp_get_num_rows(self.prob)
        cdef int ncols = glp_get_num_cols(self.prob)

        self.row_stat = np.empty((nscenarios, nrows), dtype=np.int32)
        self.col_stat = np.empty((nscenarios, ncols), dtype=np.int32)

    cdef _save_basis(self, int global_id):
        """ Save the current basis for scenario associated with global_id """
        cdef int i
        cdef int nrows = glp_get_num_rows(self.prob)
        cdef int ncols = glp_get_num_cols(self.prob)

        for i in range(nrows):
            self.row_stat[global_id, i] = glp_get_row_stat(self.prob, i+1)
        for i in range(ncols):
            self.col_stat[global_id, i] = glp_get_col_stat(self.prob, i+1)

    cdef _set_basis(self, int global_id):
        """ Set the current basis for scenario associated with global_id """
        cdef int i, nrows, ncols

        if self.is_first_solve:
            # First time solving we use the default advanced basis
            glp_adv_basis(self.prob, 0)
        else:
            # otherwise we restore basis from previous solve of this scenario
            nrows = glp_get_num_rows(self.prob)
            ncols = glp_get_num_cols(self.prob)

            for i in range(nrows):
                glp_set_row_stat(self.prob, i+1, self.row_stat[global_id, i])
            for i in range(ncols):
                glp_set_col_stat(self.prob, i+1, self.col_stat[global_id, i])

    def reset(self):
        # Resetting this triggers a crashing of a new basis in each scenario
        self.is_first_solve = True

    cpdef object solve(self, model):
        t0 = time.perf_counter()
        cdef int[:] scenario_combination
        cdef int scenario_id
        cdef ScenarioIndex scenario_index
        for scenario_index in model.scenarios.combinations:
            self._solve_scenario(model, scenario_index)
        self.stats['total'] += time.perf_counter() - t0
        # After solving this is always false
        self.is_first_solve = False

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef object _solve_scenario(self, model, ScenarioIndex scenario_index):
        cdef Node gen, load, line
        cdef Storage battery
        cdef double min_gen, min_load
        cdef double max_gen, max_load
        cdef double cost
        cdef double max_volume
        cdef double min_volume
        cdef double avail_volume
        cdef double t0
        cdef int col, row
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status, simplex_ret
        cdef cross_domain_col
        cdef list route
        cdef int node_id, indptr, nbuses
        cdef double flow
        cdef int n, m
        cdef Py_ssize_t length

        timestep = model.timestep
        cdef list buses = self.buses
        cdef list lines = self.lines
        cdef list generators = self.generators
        cdef list loads = self.loads
        cdef list batteries = self.batteries
        nbuses = len(buses)
        # update route cost

        t0 = time.perf_counter()

        # calculate the total cost of each generator and load
        for col, gen in enumerate(generators):
            cost = gen.get_cost(scenario_index)
            set_obj_coef(self.prob, self.idx_col_generation+col, cost)
        for col, load in enumerate(loads):
            cost = load.get_cost(scenario_index)
            set_obj_coef(self.prob, self.idx_col_load+col, cost)

        self.stats['objective_update'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update min/max generation & load
        for col, gen in enumerate(generators):
            # Update generation constraints
            min_gen = inf_to_dbl_max(gen.get_min_flow(scenario_index))
            if abs(min_gen) < 1e-8:
                min_gen = 0.0
            max_gen = inf_to_dbl_max(gen.get_max_flow(scenario_index))
            if abs(max_gen) < 1e-8:
                max_gen = 0.0
            set_col_bnds(self.prob, self.idx_col_generation+col, constraint_type(min_gen, max_gen),
                         min_gen, max_gen)

        # update line bounds
        for row, line in enumerate(lines):
            # NB get_min_flow is unused.
            max_load = inf_to_dbl_max(line.get_max_flow(scenario_index))
            if abs(max_load) < 1e-8:
                max_load = 0.0

            # Capacity can be +/- the max_load here (i.e. in either direction).
            set_row_bnds(self.prob, self.idx_row_line_capacity+row, constraint_type(-max_load, max_load),
                         -max_load, max_load)

        # Update load bounds
        for col, load in enumerate(loads):
            min_load = inf_to_dbl_max(load.get_min_flow(scenario_index))
            if abs(min_load) < 1e-8:
                min_load = 0.0
            max_load = inf_to_dbl_max(load.get_max_flow(scenario_index))
            if abs(max_load) < 1e-8:
                max_load = 0.0
            set_col_bnds(self.prob, self.idx_col_load+col, constraint_type(min_load, max_load),
                         min_load, max_load)

            # set_row_bnds(self.prob, self.idx_row_power_flow+col, constraint_type(min_load, max_load),
            #              min_load, max_load)

        self.stats['bounds_update_power_flow'] += time.perf_counter() - t0

        # update battery node constraint
        for row, battery in enumerate(batteries):
            max_volume = battery.get_max_volume(scenario_index)
            min_volume = battery.get_min_volume(scenario_index)

            if max_volume == min_volume:
                set_row_bnds(self.prob, self.idx_row_batteries+row, GLP_FX, 0.0, 0.0)
            else:
                avail_volume = max(battery._volume[scenario_index.global_id] - min_volume, 0.0)
                # change in battery cannot be more than the current volume or
                # result in maximum volume being exceeded
                lb = -avail_volume/timestep.days
                ub = max(max_volume - battery._volume[scenario_index.global_id], 0.0) / timestep.days

                if abs(lb) < 1e-8:
                    lb = 0.0
                if abs(ub) < 1e-8:
                    ub = 0.0
                set_row_bnds(self.prob, self.idx_row_batteries+row, constraint_type(lb, ub), lb, ub)

        t0 = time.perf_counter()

        # Apply presolve if required
        if self.use_presolve and not self.has_presolved:
            self.smcp.presolve = GLP_ON
            self.has_presolved = True
        else:
            self.smcp.presolve = GLP_OFF

        # Set the basis for this scenario
        self._set_basis(scenario_index.global_id)
        # attempt to solve the linear programme
        simplex_ret = simplex(self.prob, self.smcp)
        status = glp_get_status(self.prob)
        if (status != GLP_OPT or simplex_ret != 0) and self.retry_solve:
            # try creating a new basis and resolving
            print('Retrying solve with new basis.')
            glp_std_basis(self.prob)
            simplex_ret = simplex(self.prob, self.smcp)
            status = glp_get_status(self.prob)

        if status != GLP_OPT or simplex_ret != 0:
            # If problem is not solved. Print some debugging information and error.
            print("Simplex solve returned: {} ({})".format(simplex_status_string[simplex_ret], simplex_ret))
            print("Simplex status: {} ({})".format(status_string[status], status))
            print("Scenario ID: {}".format(scenario_index.global_id))
            print("Timestep index: {}".format(timestep._index))
            self.dump_mps(b'pywr_glpk_debug.mps')
            self.dump_lp(b'pywr_glpk_debug.lp')

            self.smcp.msg_lev = GLP_MSG_DBG
            # Retry solve with debug messages
            simplex_ret = simplex(self.prob, self.smcp)
            status = glp_get_status(self.prob)
            raise RuntimeError('Simplex solver failed with message: "{}", status: "{}".'.format(
                simplex_status_string[simplex_ret], status_string[status]))
        # Now save the basis
        self._save_basis(scenario_index.global_id)

        self.stats['lp_solve'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        for col, gen in enumerate(generators):
            g = glp_get_col_prim(self.prob, self.idx_col_generation+col)
            gen.commit(scenario_index.global_id, g)

        for col, load in enumerate(loads):
            l = glp_get_col_prim(self.prob, self.idx_col_load+col)
            load.commit(scenario_index.global_id, l)

        for row, line in enumerate(lines):
            l = glp_get_row_prim(self.prob, self.idx_row_line_capacity+row)
            line.commit(scenario_index.global_id, l)

    cpdef dump_mps(self, filename):
        glp_write_mps(self.prob, GLP_MPS_FILE, NULL, filename)

    cpdef dump_lp(self, filename):
        glp_write_lp(self.prob, NULL, filename)

    cpdef dump_glpk(self, filename):
        glp_write_prob(self.prob, 0, filename)

cdef int simplex(glp_prob *P, glp_smcp parm):
    return glp_simplex(P, &parm)


cdef set_obj_coef(glp_prob *P, int j, double coef):
    IF SOLVER_DEBUG:
        assert np.isfinite(coef)
        if abs(coef) < 1e-9:
            if abs(coef) != 0.0:
                print(j, coef)
                assert False
    glp_set_obj_coef(P, j, coef)


cdef set_row_bnds(glp_prob *P, int i, int type, double lb, double ub):
    IF SOLVER_DEBUG:
        assert np.isfinite(lb)
        assert np.isfinite(ub)
        assert lb <= ub
        if abs(lb) < 1e-9:
            if abs(lb) != 0.0:
                print(i, type, lb, ub)

                assert False
        if abs(ub) < 1e-9:
            if abs(ub) != 0.0:
                print(i, type, lb, ub)
                assert False

    glp_set_row_bnds(P, i, type, lb, ub)


cdef set_col_bnds(glp_prob *P, int i, int type, double lb, double ub):
    IF SOLVER_DEBUG:
        assert np.isfinite(lb)
        assert np.isfinite(ub)
        assert lb <= ub
    glp_set_col_bnds(P, i, type, lb, ub)


cdef set_mat_row(glp_prob *P, int i, int len, int* ind, double* val):
    IF SOLVER_DEBUG:
        cdef int j
        for j in range(len):
            assert np.isfinite(val[j+1])
            assert abs(val[j+1]) > 1e-6
            assert ind[j+1] > 0

    glp_set_mat_row(P, i, len, ind, val)
