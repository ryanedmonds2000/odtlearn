from typing import Union

from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from odtlearn import ODTL
from odtlearn.flow_opt_ms import FlowOPTMultipleSink
from odtlearn.flow_opt_ss import FlowOPTSingleSink
from odtlearn.utils.validation import (
    check_binary,
    check_columns_match,
    check_ipw,
    check_y,
    check_y_hat,
)

import numpy as np
import scipy.spatial.distance

class FlowOPT_IPW_Unobserved(FlowOPTSingleSink):
    """
    A user-facing class for learning optimal prescriptive trees with inverse probability weighting (IPW) objective under unobserved confounders.

    Parameters
    ----------
    solver : str, default="gurobi"
        The solver to use for the optimization problem. Can be either "gurobi" or "cbc".
    depth : int
        The maximum depth of the tree to be learned.
    time_limit : int, default=300
        The time limit (in seconds) for solving the optimization problem.
    num_threads : int, default=1
        The number of threads to use for solving the optimization problem.
    verbose : bool, default=False
        Whether to print verbose output during the tree learning process.

    Attributes
    ----------
    _ipw : :class:`numpy.ndarray`
        The inverse probability weights for each datapoint.

    Methods
    -------
    fit(X, y, t, ipw)
        Fit the optimal prescriptive tree using the provided data and inverse probability weights.
    _define_objective()
        Define the objective function for the optimization problem.

    Notes
    -----
    This class inherits from the :mod:`FlowOPTSingleSink <odtlearn.flow_opt_ss.FlowOPTSingleSink>` class
    and provides a user-friendly interface
    for learning optimal prescriptive trees with inverse probability weighting (IPW) objective.

    The IPW objective is used to account for potential confounding factors in observational data
    and to estimate the causal effect of treatments on outcomes. The inverse probability weights
    are computed based on the propensity scores of receiving each treatment given the observed covariates.

    The class extends the functionality of :mod:`FlowOPTSingleSink <odtlearn.flow_opt_ss.FlowOPTSingleSink>` by adding
    the `_ipw` attribute to store
    the inverse probability weights and overriding the :meth:`fit <odtlearn.flow_opt.FlowOPT_IPW.fit>` method to
    accept the IPW as an additional argument.

    The :meth:`_define_objective <odtlearn.flow_opt.FlowOPT_IPW._define_objective>` method is implemented to define
    the IPW objective function for the optimization problem.
    The objective maximizes the weighted sum of outcomes, where the weights are the product of the IPW and the
    treatment assignments.

    The class inherits the :meth:`predict <odtlearn.opt_pt.OptimalPrescriptiveTree.predict>`,
    :meth:`print_tree <odtlearn.opt_pt.OptimalPrescriptiveTree.print_tree>`,
    and :meth:`plot_tree <odtlearn.opt_pt.OptimalPrescriptiveTree.plot_tree>` methods from the
    :mod:`OptimalPrescriptiveTree <odtlearn.opt_pt.OptimalPrescriptiveTree>`
    class to make predictions and visualize the learned tree.

    Example usage:

    ```

    # Create an instance of FlowOPT_IPW
    opt_tree = FlowOPT_IPW(depth=3, time_limit=600, verbose=True)

    # Fit the optimal prescriptive tree using data and IPW
    opt_tree.fit(X, y, t, ipw)

    # Make predictions
    predictions = opt_tree.predict(X_new)

    # Plot the learned tree
    opt_tree.plot_tree()

    ```
    """

    def __init__(
        self,
        solver: str,
        depth: int = 1,
        time_limit: int = 60,
        num_threads: Union[None, int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            solver,
            depth,
            time_limit,
            num_threads,
            verbose,
        )

    
    def _dualized_variables(self) -> None:
        self._aupper = self._solver.add_vars(
            self._datapoints, 
            vtype = ODTL.CONTINUOUS,
            lb = 0,
            name="a_upper"
        )
        self._alower = self._solver.add_vars(
            self._datapoints, 
            vtype = ODTL.CONTINUOUS,
            ub = 0,
            name="a_lower"
        )
        self._beta = self._solver.add_vars(
            self._treatments,
            vtype = ODTL.CONTINUOUS,
            name="beta"
        )
        self._sigma = self._solver.add_vars(
            self._datapoints, 
            vtype = ODTL.CONTINUOUS,
            lb = 0,
            name="sigma"
        )
        self._theta = self._solver.add_vars(
            self._treatments,
            self._datapoints,
            vtype=ODTL.CONTINUOUS,
            lb=0,
            name="theta"
        )

    def _define_variables(self) -> None:
        super()._define_variables()
        self._dualized_variables()

    def _dualized_constraints(self) -> None:
        self._solver.add_constrs(
            (
                self.gamma**(-1) * self._aupper[i] + self.gamma * self._alower[i] - (self._sigma[i] / self.numagents)
                <= self._solver.quicksum(
                    self._z[i, 1] * (self._y[i]) / self._ipw[i]
                    for i in self._datapoints
                )
            )
            for i in self._datapoints
        )

        for j in self._datapoints:
            for t in self._treatments:
                for i in self._datapoints:
                    if self._t.at[i] == t:
                        self._solver.add_constrs(
                            self.distances[i, j] * self._beta[t] + self._theta[t, j] + self._sigma[i] <= 0
                        )
    def _define_constraints(self) -> None:
        super()._define_constraints()
        self._dualized_constraints()

    def _define_objectve(self) -> None:
        obj = self._solver.lin_expr(0)  
        for i in self._datapoints:
            obj += (
                self._aupper[i] * (self.gamma + ((1 - self._ipw[i]) / self.ipw[i])) 
                + self._alower[i] * (self.gamma**(-1) + ((1 - self._ipw[i]) / self.ipw[i])) 
            )      
        for t in self._treatments:
            obj += self.epsilon[t] * self._beta[t]
            for i in self._datapoints:
                obj += self._theta[t, i] / self.numagents


    def fit(
        self,
        X: Union[ndarray, DataFrame],
        t: Union[Series, ndarray],
        y: Union[Series, ndarray],
        ipw: Union[Series, ndarray],
        gamma: float,
        epsilon: ndarray,
    ) -> "FlowOPT_IPW":
        """
        Fit the FlowOPT_IPW model to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each feature should be binary (0 or 1).
        t : array-like of shape (n_samples,)
            The treatment values. An array of integers representing the treatment applied to each sample.
        y : array-like of shape (n_samples,)
            The observed outcomes upon given treatment t. An array of numbers representing the outcome for each sample.
        ipw : array-like of shape (n_samples,)
            The inverse propensity weights. An array of floats in the range (0, 1].

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If X contains non-binary values, or if X, t, y, and ipw have inconsistent numbers of samples.
        AssertionError
            If ipw contains values outside the range (0, 1].

        Notes
        -----
        This method fits the FlowOPT_IPW model using the inverse probability weighting (IPW) approach.
        It sets up the optimization problem, solves it, and stores the results.
        The IPW approach is used to account for confounding in observational data.

        Examples
        --------
        >>> from odtlearn.flow_opt import FlowOPT_IPW
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> t = np.array([0, 1, 1, 0])
        >>> y = np.array([0.5, 1.2, 0.8, 0.3])
        >>> ipw = np.array([0.8, 0.7, 0.9, 0.6])
        >>> opt = FlowOPT_IPW(depth=2, time_limit=60)
        >>> opt.fit(X, t, y, ipw)
        """



        
        # store column information and dtypes if any
        self._extract_metadata(X, y, t)

        # this function returns converted X and t but we retain metadata
        X, t = check_X_y(X, t)

        # need to check that t is discrete, and/or convert -- starts from 0 in accordance with indexing rule
        try:
            t = t.astype(int)
        except TypeError:
            print("The set of treatments must be discrete.")

        assert (
            min(t) == 0 and max(t) == len(set(t)) - 1
        ), "The set of treatments must be discrete starting from {0, 1, ...}"

        assert (
            min(epsilon) >= 0
        )
        # we also need to check on y and ipw/y_hat depending on the method chosen
        y = check_y(X, y)
        self._ipw = check_ipw(X, ipw)

        # TODO: check consistent length of epsilon with # treatments

        # Raises ValueError if there is a column that has values other than 0 or 1
        check_binary(X)

        self.numagents = np.size(self._datapoints)
        self.distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
        self.gamma = gamma
        self.epsilon = epsilon


        self._create_main_problem()
        self._solver.optimize(self._X, self, self._solver)

        self.b_value = self._solver.get_var_value(self._b, "b")
        self.w_value = self._solver.get_var_value(self._w, "w")
        self.p_value = self._solver.get_var_value(self._p, "p")

        # Return the classifier
        return self

    def predict(self, X: Union[ndarray, DataFrame]) -> ndarray:
        """
        Predict optimal treatments for samples in X using the fitted FlowOPT_IPW model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to make predictions. Each feature should be binary (0 or 1).

        Returns
        -------
        t_pred : ndarray of shape (n_samples,)
            The predicted optimal treatment for each sample in X.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If X contains non-binary values or has a different number of features than the training data.

        Notes
        -----
        This method uses the prescriptive tree learned during the fit process to recommend treatments for new samples.
        It traverses the tree for each sample in X, following the branching decisions until reaching a leaf node,
        and returns the corresponding treatment recommendation.

        Examples
        --------
        >>> from odtlearn.flow_opt import FlowOPT_IPW
        >>> import numpy as np
        >>> X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        >>> t_train = np.array([0, 1, 1, 0])
        >>> y_train = np.array([0.5, 1.2, 0.8, 0.3])
        >>> ipw_train = np.array([0.8, 0.7, 0.9, 0.6])
        >>> opt = FlowOPT_IPW(depth=2)
        >>> opt.fit(X_train, t_train, y_train, ipw_train)
        >>> X_test = np.array([[1, 1], [0, 0]])
        >>> t_pred = opt.predict(X_test)
        >>> print(t_pred)
        [1 0]
        """

        # Check if fit had been called
        check_is_fitted(self, ["b_value", "w_value", "p_value"])

        # This will again convert a pandas df to numpy array
        # but we have the column information from when we called fit
        X = check_array(X)

        check_columns_match(self._X_col_labels, X)

        return self._make_prediction(X)


