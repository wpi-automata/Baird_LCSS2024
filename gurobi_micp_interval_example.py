from ..base import STLSolver
from ...STL import LinearPredicate, NonlinearPredicate
import numpy as np
from numpy import diag_indices_from, clip, empty, inf
from scipy.ndimage import shift

import gurobipy as gp
from gurobipy import GRB, min_ #, QuadExpr

from interval import get_lu

import time

def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret

class GurobiMICPSolver(STLSolver):
    """
    Given an :class:`.STLFormula` :math:`\\varphi` and a :class:`.LinearSystem`,
    solve the optimization problem

    .. math::

        \min & -\\rho^{\\varphi}(y_0,y_1,\dots,y_T) + \sum_{t=0}^T x_t^TQx_t + u_t^TRu_t

        \\text{s.t. } & x_0 \\text{ fixed}

        & x_{t+1} = A x_t + B u_t

        & y_{t} = C x_t + D u_t

        & \\rho^{\\varphi}(y_0,y_1,\dots,y_T) \geq 0

    with Gurobi using mixed-integer convex programming. This gives a globally optimal
    solution, but may be computationally expensive for long and complex specifications.
    
    .. note::

        This class implements the algorithm described in

        Belta C, et al.
        *Formal methods for control synthesis: an optimization perspective*.
        Annual Review of Control, Robotics, and Autonomous Systems, 2019.
        https://dx.doi.org/10.1146/annurev-control-053018-023717.

    :param spec:            A tuple of :class:`.STLFormula` describing the specification, logically conjoined.
    :param sys:             A :class:`.LinearSystem` describing the system dynamics.
    :param model:           A :class:`Model` describing the system dynamics as an LTV system.
    :param x0:              A ``(n,1)`` numpy matrix describing the initial state.
    :param T:               A positive integer fixing the total number of timesteps :math:`T`.
    :param M:               (optional) A large positive scalar used to rewrite ``min`` and ``max`` as
                            mixed-integer constraints. Default is ``1000``.
    :param robustness_cost: (optional) Boolean flag for adding a linear cost to maximize
                            the robustness measure. Default is ``True``.
    :param presolve:        (optional) A boolean indicating whether to use Gurobi's
                            presolve routines. Default is ``True``.
    :param verbose:         (optional) A boolean indicating whether to print detailed
                            solver info. Default is ``True``.
    """

    def __init__(self, spec, sys, x0, T, infinite_spec=None, model=None, M=1000, robustness_cost=False, 
            hard_constraint=True, presolve=True, verbose=True, horizon=0, history=0, dT=0.25, rho_min=0.0, N=1, tube_mpc_buffer=None,
            tube_mpc_enabled=False, w=None, intervals=False, num_predicates=1):
        assert M > 0, "M should be a (large) positive scalar"
        if type(spec) is not tuple:
            spec = (spec,)
        super().__init__(spec, sys, x0, T, verbose)
        self.w=w
        self.intervals=intervals
        # self.dT=dT

        self.horizon = horizon # useful for our specific problem
        self.dynamics_model = model

        self.M = float(M)
        self.presolve = presolve
        self.num_predicates=num_predicates

        # Create a variable representing the number of future steps to require rho>0 on.
        self.N = N
        self.tubeMPCBuffer=tube_mpc_buffer
        self.tubeMPCEnabled=tube_mpc_enabled

        # Create a gurobi model to govern the optimization problem.
        self.model = gp.Model("STL_MICP")
        
        # Initialize the cost function, which will added to self.model right before solving
        self.cost = 0.0

        # Initialize a place to hold constraints
        self.dynamics_constraints = []
        self.lp_constraints = []
        self.infinite_spec_constraints = []

        self.initialization_point = None

        # Dummy start point - it's not that useful...
        #self.start_point = 2*self.horizon
        self.start_writing_ics = 0

        # Set some model parameters
        if not self.presolve:
            self.model.setParam('Presolve', 0)
        if not self.verbose:
            self.model.setParam('OutputFlag', 0)

        if self.verbose:
            print("Setting up optimization problem...")
            st = time.time()  # for computing setup time
        
        # Create optimization variables
        self.y = self.model.addMVar((self.sys.p, self.T), lb=-float('inf'), name='y')

        self.a = self.model.addMVar((self.sys.p, self.num_predicates), lb=-float('inf'), name='a')
        self.b = self.model.addMVar((1, self.num_predicates), lb=-float('inf'), name='b')

        self.x = self.model.addMVar((self.sys.n, self.T), lb=-float('inf'), name='x')
        self.u = self.model.addMVar((self.sys.m, self.T), lb=-float('inf'), name='u')
        self.s = self.model.addMVar((self.sys.m, 1), lb=-float('inf'), name='s')
        self.rho = self.model.addMVar((len(spec), self.T), name='rho', lb=rho_min) # lb sets minimum robustness
        self.ξ = self.model.addMVar((len(spec), self.T), name='ξ', lb=0.0) # slack variable for the robustness
        
        # I need to create an optimization variable for each of the predicates.


        # Create the Pre vector.
        self.infinite_spec=None; self.Pre=None
        if infinite_spec is not None:
            self.infinite_spec = infinite_spec
            # self.Pre = np.ones((infinite_spec.delay, 1)) * np.inf # all infinite, and it will be updated recursively like a controller.
            # self.Pre = np.ones((history, 1)) * self.M
            self.Pre = self.model.addMVar((history, 1), lb=-float('inf'), name='Pre')
            self.alpha = np.ones((history, 1)) * self.M
            self.history = history
            self.pre_constraints = [] # for updating element-wise the upper bound on the Pre variables.
            # print(self.Pre)

        # Add cost and constraints to the optimization problem
        self.AddDynamicsConstraints() # Update: this is called only when first initializing the problem.
        self.AddSTLConstraints()
        if hard_constraint:
            self.AddRobustnessConstraint(rho_min=rho_min)
        if robustness_cost:
            self.AddRobustnessCost()
            self.AddSoftRobustnessConstraint()

        # self.model.setParam('TimeLimit', dT * 0.9)

        if self.verbose:
            print(f"Setup complete in {time.time()-st} seconds.")

    # Note: this function is currently unused, as this approach
    # increased computation time.
    def updateStart(self, u_start=None):
        # Updates the solver start variable with u_last
        # Update specifically self.u
        if u_start is not None:
            print('updating the u.Start vector')
            self.u.Start = u_start
        else: # clear out the variable.
            self.u.Start = GRB.UNDEFINED

    def updateModel(self, sys, x0, i, u_hat, initialization_point=None):
        # Updates the Gurobi model with new system information
        # (remember: we handle LTV systems this way)

        # i is the "current time step." As far as we are concerned for the model, we
        # want to start at (history - i) + horizon.
        self.start_writing_ics = max(0, self.horizon-i)
        # so, we will ideally write the last 1,2,...,self.horizon time steps of x0.
        # this is the index to start writing x0 into the model at.
        #max(self.horizon, 2*self.horizon - i)
        self.x0 = x0
        self.sys = sys

        if initialization_point is not None:
            self.initialization_point = initialization_point # used for the dynamics constraints with a NL model.

        self.RemoveDynamicsConstraints()
        # self.AddDynamicsConstraints()
        self.AddHistoricDynamicsConstraints()

        if self.infinite_spec is not None:
            self.RemoveInfiniteSpecConstraints()
            self.AddInfiniteSpecConstraints(self.horizon, self.z_spec)
        
        self.RemoveLPConstraints()
        self.AddLPConstraints(self.horizon-1, u_hat)

    def AddControlBounds(self, u_min, u_max):
        # if type(u_max) in [float, int]:
        for t in range(self.T): # range(self.start_point, self.T)
            self.model.addConstr( u_min <= self.u[:,t] )
            self.model.addConstr( self.u[:,t] <= u_max )
        # else:
        #     print('[WARN] bounds must be float or int.')
    def AddControlBoundsTubeMPC(self, u_min, u_max):
        assert( len(u_min) == self.N )
        for t in range(self.T): # Start enforcing TubeMPC input constraints at self.horizon-1.
            t_a = min(t - (self.horizon - 1), self.N-1) # use the last value for everything past t + self.N-1
            t_a = max(0, t_a) # use the first value before self.horizon - 1.
            self.model.addConstr( u_min[t_a] <= self.u[:, t] )
            self.model.addConstr( self.u[:,t] <= u_max[t_a] )

    def AddStateBounds(self, x_min, x_max):
        for t in range(self.T):
            self.model.addConstr( x_min <= self.x[:,t] )
            self.model.addConstr( self.x[:,t] <= x_max )

    def AddQuadraticCost(self, Q, R):
        self.cost += self.x[:,0]@Q@self.x[:,0] + self.u[:,0]@R@self.u[:,0]
        for t in range(1,self.T):
            self.cost += self.x[:,t]@Q@self.x[:,t] + self.u[:,t]@R@self.u[:,t]

        if self.verbose:
            print(type(self.cost))

    def AddLPCost(self):
        self.cost += (np.ones((1, self.sys.m)) @ self.s)[0,0]

    def AddLPConstraints(self, i, u_hat):
        # Add two constraints with matrices appropriately
        self.lp_constraints.append( self.u[:, i:i+1] - self.s <= u_hat )
        self.lp_constraints.append( -self.s - self.u[:, i:i+1] <= -u_hat )

        self.lp_constraints = self.model.addConstrs(_ for _ in self.lp_constraints)
    
    # Fix c in R
    # TODO: encode this efficiently!
    def AddInputOneNormConstraint(self, c):
        # Add another dummy variable.
        # d = self.model.addMVar((self.sys.m, self.T), vtype=GRB.CONTINUOUS)
        # # (self.sys.m, self.T), lb=-float('inf')
        # for j in range(self.T): # For each self.u[:, j]
        #     # enforce the arguments are <= d, as is their sum
        #     self.model.addConstr( d[j] >= sum(self.u[j,:] ) )
        #     self.model.addConstr( d[j] >= -sum(self.u[j,:]) )
        for i in range(self.T):
            self.model.addConstr( c >= self.u[:, i]@self.u[:, i] )

    def RemoveLPConstraints(self):
        if len(self.lp_constraints) > 0:
            self.model.remove(self.lp_constraints)
        self.lp_constraints = []
        return

    def AddControlCost(self, u_hat):
        for t in range(self.T):
            self.cost += (u_hat - self.u[:, t]) * (self.T - t)
    
    def AddRobustnessCost(self):
        for j in range(len(self.spec)):
            for t in range(self.T):# - self.horizon): # range(self.N + self.horizon):
                if j == 0:
                    gamma_t = 1000
                else:
                    gamma_t = 500
                if t < self.horizon:
                    gamma_t *= 1
                else:
                    pass
                    # gamma_t *= (0.5 ** (t - self.horizon))
                    # gamma_t = 0.5 ** (t - self.horizon)
                self.cost += gamma_t * self.ξ[j,t]
            # self.cost -= 1*self.rho

    # is the +1 necessary?
    def AddRecursiveFeasibilityConstraint(self, polytope, idx=0):
        if type(polytope) is list:
            # Conjoin the polytopes using integer variables.
            n = len(polytope) # number of polytopes to join.
            z = self.model.addMVar(n, vtype=GRB.BINARY)
            total_z_to_enforce = 0
            for j in range(n):
                # print(polytope)
                poly = polytope[j][0] # polytope to work with
                idx = polytope[j][1] # offset from the end of self.x
                do_z = polytope[j][2] # whether this is an OR requirement
                if do_z:
                    for i in range(poly.p.shape[0]):
                        self.model.addConstr(poly.P[i,:]@self.x[:, self.N + self.horizon - idx] <= poly.p[i] + self.M*(1-z[j]))
                else:
                    total_z_to_enforce += 1
                    for i in range(poly.p.shape[0]):
                        self.model.addConstr(poly.P[i,:]@self.x[:, self.N + self.horizon - idx] <= poly.p[i])
            if total_z_to_enforce < n:
                self.model.addConstr(sum(z) >= total_z_to_enforce + 1)
        elif type(polytope) is tuple:
            mainPolytope = polytope[0]
            for i in range(mainPolytope.p.shape[0]):
                self.model.addConstr( mainPolytope.P[i,:]@self.x[:, self.N + self.horizon + 1] <= mainPolytope.p[i] )
            
            # For the rest: conjoin these using integer variables.
            n = len(polytope) - 1 # number of polytopes to join.
            z = self.model.addMVar(n, vtype=GRB.BINARY)
            for j in range(n):
                for i in range(polytope[j+1].p.shape[0]):
                    self.model.addConstr(polytope[j+1].P[i,:]@self.x[:, self.N + self.horizon + 1] <= polytope[j+1].p[i] + self.M*(1-z[j]))
            self.model.addConstr(sum(z) >= 1)
        else:
            for i in range(polytope.p.shape[0]):
                self.model.addConstr( polytope.P[i,:]@self.x[:, self.N + self.horizon - idx] <= polytope.p[i] )
    
    def AddPredicateIntervalConstraints(self, a_interval, b_interval):
        # Add an interval constraint on each of the values of alpha and beta.
        a_l, a_u = get_lu(a_interval)
        b_l, b_u = get_lu(b_interval)

        print(a_l)
        print(a_u)
        print(b_l)
        print(b_u)

        self.model.addConstr( self.b <= b_u )
        self.model.addConstr( self.b >= b_l )

        self.model.addConstr( self.a <= a_u )
        self.model.addConstr( self.a >= a_l )

    def AddRobustnessConstraint(self, rho_min=0.0):
        self.model.addConstr( self.rho >= rho_min )
    
    def AddSoftRobustnessConstraint(self):
        self.model.addConstr(self.rho >= -self.ξ)

    def Solve(self):
        # Set the cost function now, right before we solve.
        # This is needed since model.setObjective resets the cost.
        # print(type(self.cost))
        # print(self.cost)
        self.model.setObjective(self.cost, GRB.MINIMIZE)
        # print(self.model.getVars())
        # Do the actual solving
        self.model.optimize()
        success = None

        if self.model.status == GRB.OPTIMAL:
            if self.verbose:
                print("\nOptimal Solution Found!\n")
            x = self.x.X
            u = self.u.X
            rho = self.rho.X

            # Update the Pre vector for the Since clause using its LinearPredicate formula.
            if self.Pre is not None:
                # Calculate the predicate value of the current time step. What is the current time?
                currentPredicateValue = (self.infinite_spec.a * self.y.X[:,self.start_writing_ics] - self.infinite_spec.b) # maybe self.horizon?
                updatePreValue = min(np.min(self.alpha), currentPredicateValue) # + 1e-6
                # if updatePreValue < 0:
                #     updatePreValue -= 1
                self.alpha = np.array([shift(self.alpha[:, 0], -1, cval=updatePreValue)]).T
                if self.verbose:
                    print(f'test: {self.infinite_spec.a * self.y.X[:,self.start_writing_ics] - self.infinite_spec.b}')
                    print(f'new pre = {updatePreValue}')
                    print(f'self.alpha = {self.alpha}')
                    if updatePreValue <= 0:
                        print(f'updatePreValue == {updatePreValue}, and now it\'s negative!\n\n\n this is bad.')
                self.UpdatePreConstraints()
                # time.sleep(1)

            # Report optimal cost and robustness
            if self.verbose:
                print("Solve time: ", self.model.Runtime)
                print("Optimal robustness: ", rho)
                print("")
                # for i in range(len(self.z_specs)):
                #     print(f'Resultant z_specs[{i}]: ', self.z_specs[i].X)
            
            success = True
            objective = self.model.getObjective().getValue()
        else:
            if self.verbose:
                print(f"\nOptimization failed with status {self.model.status}.\n")
            x = None
            u = None
            rho = -np.inf
            success = False
            objective = np.inf
        
        if self.verbose:
            print(self.model.status)
        return (x,u,rho,self.model.Runtime,objective,success)
    
    def AddHistoricDynamicsConstraints(self):
        h = self.x0.shape[1] # number of historical conditions.

        # Update with historical conditions
        if self.verbose:
            print('x0')
            print(self.x0)

        for i in range(self.start_writing_ics, self.horizon):
            x0_idx = h - self.horizon + i# indexer corresponding to the last values of x0
            if self.verbose:
                print(f'initializing index {i}, x[:,{i}]')
                print(self.x0[:, x0_idx:x0_idx+1])
            
            # Eliminate C array slicing due to Vars vs MVars.
            self.dynamics_constraints.append( self.x[:, i] == self.x0[:, x0_idx] )

        self.dynamics_constraints = self.model.addConstrs((_ for _ in self.dynamics_constraints))
        # There must exist a more efficient way of doing this.

    # Note: the indices were extensively verified as of 7/9/23 to be correct.
    def AddDynamicsConstraints(self):
        # Dynamics (that are not updated with each update to the historical states)
        if self.dynamics_model is None:
            for t in range(self.horizon-1, self.T-1):
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}], y[:,{t}] using u[:,{t}]')
                if self.w is not None:
                    self.model.addConstr(
                            self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] + self.w[:,0] )
                else:
                    self.model.addConstr(
                            self.x[:,t+1] == self.sys.A@self.x[:,t] + self.sys.B@self.u[:,t] )
            for t in range(self.T):
                self.model.addConstr(
                        self.y[:,t] == self.sys.C@self.x[:,t] )
        else:
            if self.verbose:
                print('x0')
                print(self.x0[:, -1:])

            A_discrete, B_discrete = self.dynamics_model.get_discrete_model()
            
            for t in range(self.horizon-1, self.T-1):
                # indexer = t - (self.horizon - 1)
                # mI = t - (self.horizon-1)
                if self.verbose:
                    print(f'initializing index {t}, x[:,{t+1}] using u[:,{t}] and model')
                self.model.addConstr(
                        self.x[:,t+1] == A_discrete@self.x[:,t] + B_discrete@self.u[:,t] ) 
                
            for t in range(self.T):
                if self.verbose:
                    print(f'initializing index {t}, y[:,{t}] using y=C@x[:,{t}]')
                self.model.addConstr(
                        self.y[:,t] == self.dynamics_model.C@self.x[:,t] )

    def RemoveDynamicsConstraints(self):
        # remove previously added dynamics constraints.
        if len(self.dynamics_constraints) > 0:
            self.model.remove(self.dynamics_constraints)
        
        # reset the array
        self.dynamics_constraints = []

    def AddSTLConstraints(self):
        """
        Add the STL constraints

            (x,u) |= specification

        to the optimization problem, via the recursive introduction
        of binary variables for all subformulas in the specification.
        """
        # Recursively traverse the tree defined by the specification
        # to add binary variables and constraints that ensure that
        # rho is the robustness value

        #
        self.z_spec = self.model.addMVar(1, vtype=GRB.CONTINUOUS)
        for j in range(len(self.spec)):
            start = 0#max(self.x0.shape[1] - self.horizon, 0)
            temp_spec = self.spec[j].always(start, self.horizon+self.N)
            if self.verbose:
                print(f'range is {start} to {self.horizon + self.N}')
            self.AddFormulaConstraintsNotRecursive(temp_spec, self.z_spec, 0, 0, j)

        if self.infinite_spec is not None:
            self.AddInfiniteSpecConstraints(self.horizon, self.z_spec)

        self.model.addConstr( self.z_spec == 1)
        
    def UpdatePreConstraints(self):
        self.model.remove(self.pre_constraints)
        self.pre_constraints = []
        for i in range(self.history):
            self.pre_constraints.append((self.alpha[i, 0] >= self.Pre[i, 0]))
        self.pre_constraints = self.model.addConstrs((_ for _ in self.pre_constraints))

    def AddFormulaConstraintsNotRecursive(self, topLevelFormula, topZ, topT, topIdx, specIdx):
        # start with a stack data structure
        stack = []
        stack.append((topLevelFormula, topZ, topT, topIdx, 0))
        # print(topLevelFormula)
        while len(stack) > 0:
            (formula, z, t, idx, predicateIdx) = stack.pop()
            if isinstance(formula, LinearPredicate):
                if t < 0: continue # the index is invalid most likely due to a past time formula.
                self.model.addConstr( self.a[:,predicateIdx] @ self.y[:,t:t+1] - self.b[:,predicateIdx] + (1-z)*self.M  >= self.rho[specIdx, t] )
                b = self.model.addMVar(1,vtype=GRB.BINARY)
                self.model.addConstr(z == b)
            elif isinstance(formula, NonlinearPredicate):
                raise TypeError("Mixed integer programming does not support nonlinear predicates")
            else:
                if formula.combination_type == "and":
                    for i, subformula in enumerate(formula.subformula_list):
                        t_sub = formula.timesteps[i]
                        # Check if this formula is past time.
                        if formula.pre_flag and t+t_sub >= 0:
                            if self.verbose:
                                print(f'pre_flag is true.')
                                print(formula)
                                print(subformula)
                            z_2 = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                            b_2 = self.model.addMVar(1,vtype=GRB.BINARY)
                            if t+t_sub < self.history:
                                self.model.addConstr(self.Pre[t + t_sub, 0] + (1-z_2) * self.M >= self.rho[specIdx, t+t_sub])
                            else:
                                # Require that the robustness be bounded above by whatever happened history seconds ago. And history-1 seconds ago. etc.
                                self.model.addConstr(subformula.a * self.y[:, t+t_sub-self.history] - subformula.b + (1-z_2)*self.M >= self.rho[specIdx, t+t_sub-self.history])
                            
                            self.model.addConstr(z_2 == b_2)
                            self.model.addConstr( z <= z_2 )
                        else:
                            z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                            stack.append((subformula, z_sub, t+t_sub, idx, predicateIdx))
                            self.model.addConstr( z <= z_sub )
                        if formula.is_conjunction_or_disjunction():
                            predicateIdx += 1 # Not quite.
                else:  # combination_type == "or":
                    z_subs = []
                    for i, subformula in enumerate(formula.subformula_list):
                        # print(subformula)
                        # print(t_sub)
                        # if t+t_sub >= 0:
                        z_sub = self.model.addMVar(1,vtype=GRB.CONTINUOUS)
                        z_subs.append(z_sub)
                        t_sub = formula.timesteps[i]
                        # print(f't_sub = {t_sub}')
                        stack.append((subformula, z_sub, t+t_sub, idx, predicateIdx)) # Negative times are handled by the predicate.
                        if formula.is_conjunction_or_disjunction():
                            predicateIdx += 1 # Not quite.

                    self.model.addConstr( z <= sum(z_subs) )
