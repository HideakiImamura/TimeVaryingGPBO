import GPyOpt
from GPyOpt.core import BO
from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.space import Design_space
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.core.task.cost import CostModel
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.duplicate_manager import DuplicateManager

from optuna.samplers.gp.arg_manager import FlexibleArgumentsManager
from optuna.samplers.gp.optimizer import FlexibleAcquisitionOptimizer


# (TODO) Caution!!! This BO impl cannnot deal with Context  variable to avoid the conflict with CTV sampler


class FlexibleBayesianOptimization(BO):

    def __init__(
            self,
            f,
            domain=None,
            constraints=None,
            cost_withGradients=None,
            model_type='GP',
            X=None,
            Y=None,
            initial_design_numdata=5,
            initial_design_type='random',
            acquisition_type='EI',
            normalize_Y=True,
            exact_feval=False,
            acquisition_optimizer_type='lbfgs',
            model_update_interval=1,
            evaluator_type='sequential',
            batch_size=1,
            num_cores=1,
            verbosity=False,
            verbosity_model=False,
            maximize=False,
            de_duplication=False,
            **kwargs
    ):
        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kwargs
        self.problem_config = FlexibleArgumentsManager(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)
        # hoge = self.kwargs.get('domain_without_time', [])
        # import numpy as np
        # print("domain_without_time = {}".format(np.asarray(hoge)))
        # print("domain_with_time    = {}".format(np.asarray(self.domain)))
        self.space_without_time = Design_space(self.kwargs.get('domain_without_time', []), self.constraints)

        # --- CHOOSE objective function
        self.maximize = maximize
        if 'objective_name' in kwargs:
            self.objective_name = kwargs['objective_name']
        else:
            self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = SingleObjective(self.f, self.batch_size, self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type.
        # If an instance of a GPyOpt model is passed (possibly user defined),
        # it is used.
        self.model_type = model_type
        self.exact_feval = exact_feval  # note that this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y

        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print('Using a model defined by the used.')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type

        # This states how the discrete variables are handled (exact search or rounding)
        self.acquisition_optimizer_type = acquisition_optimizer_type
        if 'is_continuous_time_varying' in self.kwargs and self.kwargs['is_continuous_time_varying']:
            print("is continuous Varying!")
            self.acquisition_optimizer = FlexibleAcquisitionOptimizer(
                self.space_without_time,
                self.acquisition_optimizer_type,
                model=self.model,
            )  # more arguments may come here
        else:
            self.acquisition_optimizer = FlexibleAcquisitionOptimizer(
                self.space,
                self.acquisition_optimizer_type,
                model=self.model
            )

        # --- CHOOSE acquisition function.
        # If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type
        if 'acquisition' in self.kwargs:
            if isinstance(kwargs['acquisition'], GPyOpt.acquisitions.AcquisitionBase):
                self.acquisition = kwargs['acquisition']
                self.acquisition_type = 'User defined acquisition used.'
                print('Using an acquisition defined by the used.')
            else:
                self.acquisition = self._acquisition_chooser()
        else:
            self.acquisition = self.acquisition = self._acquisition_chooser()

        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(FlexibleBayesianOptimization, self).__init__(
            model=self.model,
            space=self.space,
            objective=self.objective,
            acquisition=self.acquisition,
            evaluator=self.evaluator,
            X_init=self.X,
            Y_init=self.Y,
            cost=self.cost,
            normalize_Y=self.normalize_Y,
            model_update_interval=self.model_update_interval,
            de_duplication=self.de_duplication)

    def _model_chooser(self):
        return self.problem_config.model_creator(
            self.model_type,
            self.exact_feval,
            self.space)

    def _acquisition_chooser(self):
        return self.problem_config.acquisition_creator(
            self.acquisition_type,
            self.model,
            self.space,
            self.acquisition_optimizer,
            self.cost.cost_withGradients,
            **self.kwargs
        )

    def _evaluator_chooser(self):
        return self.problem_config.evaluator_creator(
            self.evaluator_type,
            self.acquisition,
            self.batch_size,
            self.model_type,
            self.model,
            self.space,
            self.acquisition_optimizer)

    def _init_design_chooser(self):
        """
        Initializes the choice of X and Y based on the selected initial design and number of points selected.
        """

        # If objective function was not provided, we require some initial sample data
        if self.f is None and (self.X is None or self.Y is None):
            raise InvalidConfigError(
                "Initial data for both X and Y is required when objective function is not provided")

        # Case 1:
        if self.X is None:
            self.X = initial_design(
                self.initial_design_type,
                self.space,
                self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)
        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)

    def _sign(self, f):
        if self.maximize:
            f_copy = f

            def f(x): return -f_copy(x)
        return f

    def _delete_time_param_from_space(self):
        import pprint
        space = self.space
        pprint.pprint(space)
        return space

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        """
        Computes the location of the new evaluation (optimizes the acquisition in the standard case).
        :param pending_zipped_X: matrix of input configurations that are in a pending state (i.e., do not have an evaluation yet).
        :param ignored_zipped_X: matrix of input configurations that the user black-lists, i.e., those configurations will not be suggested again.
        :return:
        """

        ## --- Update the context if any
        # self.acquisition.optimizer.context_manager = ContextManager(self.space, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            duplicate_manager = DuplicateManager(space=self.space, zipped_X=self.X, pending_zipped_X=pending_zipped_X,
                                                 ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        ### We zip the value in case there are categorical variables

        if 'is_continuous_time_varying' in self.kwargs and self.kwargs['is_continuous_time_varying']:
            ret = self.space_without_time.zip_inputs(self.evaluator.compute_batch(
                duplicate_manager=duplicate_manager,
                context_manager=self.acquisition.optimizer.context_manager))
        else:
            ret = self.space.zip_inputs(self.evaluator.compute_batch(
                duplicate_manager=duplicate_manager,
                context_manager=self.acquisition.optimizer.context_manager))

        return ret
