# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.optimization.optimizer import OptLbfgs, OptDirect, OptCma, apply_optimizer, choose_optimizer
from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design import initial_design
from GPyOpt.core.errors import FullyExploredOptimizationDomainError
import numpy as np

max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompsom_sampling"
sobol_design_type = "sobol"
random_design_type = "random"


class FlexibleAcquisitionOptimizer(object):
    """
    General class for acquisition optimizers defined in domains with mix of discrete, continuous, bandit variables
    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    def __init__(self, space, optimizer='lbfgs', **kwargs):

        self.space = space
        self.optimizer_name = optimizer
        self.kwargs = kwargs

        ### -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        if 'anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)
        # print("In FlexibleAcquisitionOptimizer.__init__:")
        # print("     self.context_manager.nocontext_index    : ", self.context_manager.noncontext_index)
        # print("     self.context_manager.nocontext_bounds   : ", self.context_manager.noncontext_bounds)
        # print("     self.context_manager.nocontext_index_obj: ", self.context_manager.nocontext_index_obj)

    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None):
        """
        Optimizes the input function.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)
        # print("In FlexibleAcquisitionOptimizer.optimize:")
        # print("     self.context_manager.nocontext_index    : ", self.context_manager.noncontext_index)
        # print("     self.context_manager.nocontext_bounds   : ", self.context_manager.noncontext_bounds)
        # print("     self.context_manager.nocontext_index_obj: ", self.context_manager.nocontext_index_obj)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)

        ## -- Select the anchor points (with context)
        # print("In FlexibleAcquisitionOptimizer.optimize:")
        # print("     self.context_manager.nocontext_index    : ", self.context_manager.noncontext_index)
        # print("     self.context_manager.nocontext_bounds   : ", self.context_manager.noncontext_bounds)
        # print("     self.context_manager.nocontext_index_obj: ", self.context_manager.nocontext_index_obj)
        anchor_points = anchor_points_generator.get(duplicate_manager=duplicate_manager,
                                                    context_manager=self.context_manager)

        ## --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the context)
        optimized_points = [
            apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager,
                            context_manager=self.context_manager, space=self.space) for a in anchor_points]
        x_min, fx_min = min(optimized_points, key=lambda t: t[1])

        # x_min, fx_min = min([apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points], key=lambda t:t[1])

        return x_min, fx_min


class ContextManager(object):
    """
    class to handle the context variable in the optimizer
    :param space: design space class from GPyOpt.
    :param context: dictionary of variables and their contex values
    """

    def __init__(self, space, context=None):
        self.space = space
        self.all_index = list(range(space.model_dimensionality))
        self.all_index_obj = list(range(len(self.space.config_space_expanded)))
        self.context_index = []
        self.context_value = []
        self.context_index_obj = []
        self.nocontext_index_obj = self.all_index_obj
        self.noncontext_bounds = self.space.get_bounds()[:]
        self.noncontext_index = self.all_index[:]

        # print("In ContextManager: ")
        # print("     all_index    : ", self.all_index)
        # print("     all_index_obj: ", self.all_index_obj)
        # print("     context      : ", context)

        if context is not None:

            ## -- Update new context
            for context_variable in context.keys():
                variable = self.space.find_variable(context_variable)
                self.context_index += variable.index_in_model
                self.context_index_obj += variable.index_in_objective
                self.context_value += variable.objective_to_model(context[context_variable])

            ## --- Get bounds and index for non context
            self.noncontext_index = [idx for idx in self.all_index if idx not in self.context_index]
            self.noncontext_bounds = [self.noncontext_bounds[idx] for idx in self.noncontext_index]

            ## update non context index in objective
            self.nocontext_index_obj = [idx for idx in self.all_index_obj if idx not in self.context_index_obj]

        # print("     non_c_index    : ", self.noncontext_index)
        # print("     non_c_bound    : ", self.noncontext_bounds)
        # print("     non_c_indoj    : ", self.nocontext_index_obj)

    def _expand_vector(self, x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        :param x: input vector to be expanded by adding the context values
        '''
        x = np.atleast_2d(x)
        x_expanded = np.zeros((x.shape[0], self.space.model_dimensionality))
        x_expanded[:, np.array(self.noncontext_index).astype(int)] = x
        x_expanded[:, np.array(self.context_index).astype(int)] = self.context_value
        return x_expanded


class AnchorPointsGenerator(object):

    def __init__(self, space, design_type, num_samples):
        self.space = space
        self.design_type = design_type
        self.num_samples = num_samples

    def get_anchor_point_scores(self, X):
        raise NotImplementedError("get_anchor_point_scores is not implemented in the parent class.")

    def get(self, num_anchor=5, duplicate_manager=None, unique=False, context_manager=None):

        ## --- We use the context handler to remove duplicates only over the non-context variables
        if context_manager and not self.space._has_bandit():
            # print("In AnchorPointsGenerator: ")
            # print("         space.config_space_expanded        : ", self.space.config_space_expanded)
            # print("         context_manager.nocontext_index_obj: ", context_manager.noncontext_bounds)
            space_configuration_without_context = [self.space.config_space_expanded[idx] for idx in context_manager.nocontext_index_obj]
            space = Design_space(space_configuration_without_context, context_manager.space.constraints)
            add_context = lambda x : context_manager._expand_vector(x)
        else:
            space = self.space
            add_context = lambda x: x

        ## --- Generate initial design
        X = initial_design(self.design_type, space, self.num_samples)

        if unique:
            sorted_design = sorted(list({tuple(x) for x in X}))
            X = space.unzip_inputs(np.vstack(sorted_design))
        else:
            X = space.unzip_inputs(X)

        ## --- Add context variables
        X = add_context(X)

        if duplicate_manager:
            is_duplicate = duplicate_manager.is_unzipped_x_duplicate
        else:
            # In absence of duplicate manager, we never detect duplicates
            is_duplicate = lambda _ : False

        non_duplicate_anchor_point_indexes = [index for index, x in enumerate(X) if not is_duplicate(x)]

        if not non_duplicate_anchor_point_indexes:
            raise FullyExploredOptimizationDomainError("No anchor points could be generated ({} used samples, {} requested anchor points).".format(self.num_samples,num_anchor))

        if len(non_duplicate_anchor_point_indexes) < num_anchor:
            # Since logging has not been setup yet, I do not know how to express warnings...I am using standard print for now.
            print("Warning: expecting {} anchor points, only {} available.".format(num_anchor, len(non_duplicate_anchor_point_indexes)))

        X = X[non_duplicate_anchor_point_indexes,:]

        scores = self.get_anchor_point_scores(X)

        anchor_points = X[np.argsort(scores)[:min(len(scores),num_anchor)], :]

        return anchor_points


class ThompsonSamplingAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, model, num_samples=25000):
        '''
        From and initial design, it selects the location using (marginal) Thompson sampling
        using the predictive distribution of a model
        model: NOTE THAT THE MODEL HERE IS is a GPyOpt model: returns mean and standard deviation
        '''
        super(ThompsonSamplingAnchorPointsGenerator, self).__init__(space, design_type, num_samples)
        self.model = model

    def get_anchor_point_scores(self, X):

        posterior_means, posterior_stds = self.model.predict(X)

        return np.array([np.random.normal(m, s) for m, s in zip(posterior_means, posterior_stds)]).flatten()


class ObjectiveAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, objective, num_samples=1000):
        '''
        From an initial design, it selects the locations with the minimum value according to some objective.
        :param model_space: set to true when the samples need to be obtained for the input domain of the model
        '''
        super(ObjectiveAnchorPointsGenerator, self).__init__(space, design_type, num_samples)
        self.objective = objective

    def get_anchor_point_scores(self, X):

        return self.objective(X).flatten()


class RandomAnchorPointsGenerator(AnchorPointsGenerator):

    def __init__(self, space, design_type, num_samples=10000):
        '''
        From an initial design, it selects the locations randomly, according to the specified design_type generation scheme.
        :param model_space: set to true when the samples need to be obtained for the input domain of the model
        '''
        super(RandomAnchorPointsGenerator, self).__init__(space, design_type, num_samples)

    def get_anchor_point_scores(self, X):

        return range(X.shape[0])