import numpy as np
import GPy
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Union  # NOQA

from optuna import distributions  # NOQA
from optuna.distributions import BaseDistribution  # NOQA
from optuna.samplers import base  # NOQA
from optuna.samplers import random  # NOQA
from optuna.samplers.gp import bo as original_bo  # NOQA
from optuna.storages.base import BaseStorage  # NOQA


class DTVGPSampler(base.BaseSampler):

    def __init__(
            self,
            epsilon=0.01,  # type: float
            time_kernel='Bogunovic',  # type: str
            space_kernel='Matern52',  # type: str
            consider_ARD=False,  # type: bool
            acquisition_type='Adaptive_LCB_MCMC',  # type: str
            n_startup_trials=10,  # type: int
            seed=None,  # type: Optional[int]
            **kwargs  # type: Any
    ):
        # type: (...) -> None
        self.domains = {}
        self.funcs = {}
        self.epsilon = epsilon
        self.time_kernel = time_kernel
        self.space_kernel = space_kernel
        self.consider_ARD = consider_ARD
        self.acquisition_type = acquisition_type
        self.n_startup_trials = n_startup_trials
        self.seed = seed

        self.kwargs = kwargs

        self.rng = np.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> Union[float, Dict]

        observation_pairs = np.asarray(storage.get_trial_param_result_pairs(study_id, param_name))
        n = len(observation_pairs)
        print(n)

        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        x_step = np.asarray([list(d.values()) + [i + 1] for i, d in enumerate(observation_pairs[:, 0])])
        y_step = observation_pairs[:, 1][:, None]

        if isinstance(param_distribution, distributions.JointDistribution):
            if param_name not in self.domains:
                self.domains[param_name], self.funcs[param_name] = self._make_domain(
                    param_distribution.distributions_list)

                time_string = DTVGPSampler._make_time_string(self.domains[param_name])
                self.domains[param_name].append({'name': time_string,
                                                 'type': 'discrete',
                                                 'domain': tuple(np.arange(n + 1) + 1)})

            input_dim = x_step.shape[1] - 1

            kern = self.kernel_chooser(input_dim)
            bo_step = original_bo.FlexibleBayesianOptimization(f=None,
                                                               domain=self.domains[param_name],
                                                               X=x_step,
                                                               Y=y_step,
                                                               model_type='GP_MCMC',
                                                               acquisition_type=self.acquisition_type,
                                                               kernel=kern,
                                                               n_data=n,
                                                               **self.kwargs)
            print(self.domains[param_name][-1]['name'])
            x_next = bo_step.suggest_next_locations(context={self.domains[param_name][-1]['name']: n + 1})[0]

            returned_params = {}
            for i, d in enumerate(param_distribution.distributions_list):
                returned_params[d.name] = self.funcs[param_name][d.name](x_next[i])
            return returned_params
        else:
            distribution_list = [distributions.JointDistribution.__name__]
            raise NotImplementedError("The distribution {} is not implemented for GPSampler. "
                                      "The parameter distribution should be one of the {}"
                                      .format(param_distribution, distribution_list))

    def _make_domain(self, distributions_list):
        # type: (List[distributions.ElementOfDistributionsList]) -> Tuple[List[Dict], Dict[Callable]]
        domain = []
        func = {}
        for d in distributions_list:
            name = d.name
            if isinstance(d.dist, distributions.UniformDistribution):
                low = d.dist.low
                high = d.dist.high
                func[name] = lambda x: x
            elif isinstance(d.dist, distributions.LogUniformDistribution):
                low = np.log(d.dist.low)
                high = np.log(d.dist.high)
                func[name] = np.exp
            else:
                distributions_list = [distributions.UniformDistribution.__name__,
                                      distributions.LogUniformDistribution.__name__]
                raise NotImplementedError("The distribution {} is not implemented. "
                                          "The distribution for JointDistribution should be one of the {}"
                                          .format(d.dist, distributions_list))
            domain.append({'name': name, 'type': 'continuous', 'domain': (low, high)})
        return domain, func

    def kernel_chooser(self, input_dim):
        # type: (int) -> GPy.kern.Kern
        if self.space_kernel == 'Exponential' or self.space_kernel == 'Exp':
            space_kern = GPy.kern.Exponential(
                input_dim,
                ARD=self.consider_ARD,
                active_dims=[i for i in range(input_dim)]
            )
        elif self.space_kernel == 'SquaredExponential' or self.space_kernel == 'SE' or self.space_kernel == 'RBF':
            space_kern = GPy.kern.RBF(
                input_dim,
                ARD=self.consider_ARD,
                active_dims=[i for i in range(input_dim)])
        elif self.space_kernel == "Matern52":
            space_kern = GPy.kern.Matern52(
                input_dim,
                ARD=self.consider_ARD,
                active_dims=[i for i in range(input_dim)])
        else:
            kernels_list = ["Matern52",
                            'SquaredExponential',
                            'SE',
                            'RBF',
                            'Exponential',
                            'Exp']
            raise NotImplementedError("The kernel of GPSampler should be one of the ",
                                      "{}.".format(kernels_list))

        if self.time_kernel == 'Bogunovic':
            time_kern = GPy.kern.Exponential(
                1,
                lengthscale=1. / (0.5 * np.log(1. / (1. - self.epsilon))),
                active_dims=[input_dim]
            )
        elif self.time_kernel == 'Exponential' or self.time_kernel == 'Exp':
            time_kern = GPy.kern.Exponential(
                1,
                active_dims=[input_dim]
            )
        elif self.time_kernel == 'SquaredExponential' or self.time_kernel == 'SE' or self.time_kernel == 'RBF':
            time_kern = GPy.kern.RBF(
                1,
                active_dims=input_dim)
        elif self.time_kernel == "Matern52":
            time_kern = GPy.kern.Matern52(
                1,
                active_dims=[input_dim])
        else:
            kernels_list = ["Matern52",
                            'SquaredExponential',
                            'SE',
                            'RBF',
                            'Exponential',
                            'Exp',
                            'Bogunovic']
            raise NotImplementedError("The kernel of GPSampler should be one of the ",
                                      "{}.".format(kernels_list))
        kern = space_kern * time_kern
        return kern

    @classmethod
    def _make_time_string(cls, dicts_list):
        ans = ''
        for d in dicts_list:
            ans = ans + d['name']
        return ans
