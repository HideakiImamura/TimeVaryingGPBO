import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, \
    AcquisitionLCB_MCMC, AcquisitionLP
from GPyOpt.core.task.space import Design_space
from GPyOpt.models import BOModel
from GPyOpt.optimization.optimizer import Optimizer
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA

from datetime import datetime


def default_exploration_weight(x, **kwargs):
    c1 = kwargs.get('c1', 0.8)
    c2 = kwargs.get('c2', 4.)
    n = kwargs.get('n_data', 100)
    return c1 * np.log(c2 * n)


def exploration_weight_for_ctv(x, **kwargs):
    c1 = kwargs.get('c1', 0.8)
    c2 = kwargs.get('c2', 4.)
    n = kwargs.get('n_data', 100)
    t = x[:, -1]
    return (t * c1 * np.log(c2 * n)).reshape((t.shape[0], 1))


class AcquisitionLCBforCTV(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(
            self,
            model,  # type: BOModel
            space,  # type: Design_space
            optimizer=None,  # type: Optional[Optimizer]
            cost_withGradients=None,  # type: Optional[Callable]
            **kwargs  # type: Dict
    ):
        # type: (...) -> None

        self.optimizer = optimizer
        self.kwargs = kwargs
        super(AcquisitionLCBforCTV, self).__init__(model, space, optimizer)
        self.exploration_weight = kwargs.get('exploration_weight', exploration_weight_for_ctv)

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq(self, x):
        # type: (np.array) -> float

        w = self.exploration_weight(x, **self.kwargs)
        m, s = self.model.predict(x[:, :-1])
        f_acqu = -m + w * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        # type: (np.array) -> Tuple[float, float]
        w = self.exploration_weight(x, **self.kwargs)
        m, s, dmdx, dsdx = self.model.predict_withGradients(x[:, :-1])
        f_acqu = -m + w * s
        df_acqu = -dmdx + w * dsdx
        return f_acqu, df_acqu

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        # TODO(imamura): What should we do?
        raise NotImplementedError()


class AcquisitionLCBforCTV_MCMC(AcquisitionLCBforCTV):
    analytical_gradient_prediction = True

    def __init__(
            self,
            model,  # type: BOModel
            space,  # type: Design_space
            optimizer=None,  # type: Optional[Optimizer]
            cost_withGradients=None,  # type: Optional[Callable]
            **kwargs  # type: Dict
    ):
        # type: (...) -> None

        super(AcquisitionLCBforCTV_MCMC, self).__init__(
            model, space, optimizer, cost_withGradients, **kwargs)

    def _compute_acq(self, x):
        # type: (np.array) -> float

        w = self.exploration_weight(x, **self.kwargs)
        ms, ss = self.model.predict(x[:, :-1])
        f_acqu = 0.
        for m, s in zip(ms, ss):
            f_acqu += -m + w * s
        ret = f_acqu / len(ms)
        return ret

    def _compute_acq_withGradients(self, x):
        # type: (np.array) -> Tuple[float, float]
        w = self.exploration_weight(x, **self.kwargs)
        ms, ss, dmdxs, dsdxs = self.model.predict_withGradients(x[:, :-1])
        f_acqu = None
        df_acqu = None
        for m, s, dmdx, dsdx in zip(ms, ss, dmdxs, dsdxs):
            f = -m + w * s
            df = -dmdx + w * dsdx
            if f_acqu is None:
                f_acqu = f
                df_acqu = df
            else:
                f_acqu += f
                df_acqu += df
        return f_acqu / len(ms), df_acqu / len(ms)

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        # TODO(imamura): What should we do?
        raise NotImplementedError()


class AcquisitionLCBwithAdaptiveExplorationWeight(AcquisitionBase):
    analytical_gradient_prediction = True

    def __init__(
            self,
            model,  # type: BOModel
            space,  # type: Design_space
            optimizer=None,  # type: Optional[Optimizer]
            cost_withGradients=None,  # type: Optional[Callable]
            **kwargs  # type: Dict
    ):
        # type: (...) -> None

        self.optimizer = optimizer
        self.kwargs = kwargs
        super(AcquisitionLCBwithAdaptiveExplorationWeight, self).__init__(model, space, optimizer)
        self.exploration_weight = kwargs.get('exploration_weight', default_exploration_weight)

        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq(self, x):
        # type: (np.array) -> float

        w = self.exploration_weight(x, **self.kwargs)
        m, s = self.model.predict(x)
        f_acqu = -m + w * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        # type: (np.array) -> Tuple[float, float]
        w = self.exploration_weight(x, **self.kwargs)
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + w * s
        df_acqu = -dmdx + w * dsdx
        return f_acqu, df_acqu

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        # TODO(imamura): What should we do?
        raise NotImplementedError()


class AcquisitionLCBwithAdaptiveExplorationWeight_MCMC(AcquisitionLCBwithAdaptiveExplorationWeight):
    analytical_gradient_prediction = True

    def __init__(
            self,
            model,  # type: BOModel
            space,  # type: Design_space
            optimizer=None,  # type: Optional[Optimizer]
            cost_withGradients=None,  # type: Optional[Callable]
            **kwargs  # type: Dict
    ):
        # type: (...) -> None

        self.optimizer = optimizer
        super(AcquisitionLCBwithAdaptiveExplorationWeight_MCMC, self).__init__(
            model, space, optimizer, cost_withGradients, **kwargs)

    def _compute_acq(self, x):
        # type: (np.array) -> float

        w = self.exploration_weight(x)
        ms, ss = self.model.predict(x)
        f_acqu = 0.
        for m, s in zip(ms, ss):
            f_acqu += -m + w * s
        return f_acqu / len(ms)

    def _compute_acq_withGradients(self, x):
        # type: (np.array) -> Tuple[float, float]
        w = self.exploration_weight(x)
        ms, ss, dmdxs, dsdxs = self.model.predict_withGradients(x)
        f_acqu = None
        df_acqu = None
        for m, s, dmdx, dsdx in zip(ms, ss, dmdxs, dsdxs):
            f = -m + w * s
            df = -dmdx + w * dsdx
            if f_acqu is None:
                f_acqu = f
                df_acqu = df
            else:
                f_acqu += f
                df_acqu += df
        return f_acqu / len(ms), df_acqu / len(ms)

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        # TODO(imamura): What should we do?
        raise NotImplementedError()


class AcquisitionContinuousTimeVarying(AcquisitionBase):

    def __init__(
            self,
            model,
            space,
            optimizer=None,
            cost_withGradients=None,
            **kwargs
    ):
        super(AcquisitionContinuousTimeVarying, self).__init__(model, space, optimizer)
        self.optimizer = optimizer
        self.base_acquisition_type = kwargs.get('base_acquisition_type', 'LCBforCTV')
        self.model_g = kwargs.get('model_g', None)
        self.tau_n = kwargs.get('tau_n', None)
        self.n_sampling_points = kwargs.get('n_sampling_points', 30)
        self.is_delta = kwargs.get('is_delta', False)
        if self.model_g is None or self.tau_n is None:
            raise ValueError('model_g and tau_n should not be None.')

        if self.base_acquisition_type is None or self.base_acquisition_type == 'LCBforCTV':
            self.base_acq = AcquisitionLCBforCTV(
                self.model,
                self.space,
                self.optimizer,
                None,
                **kwargs
            )
        else:
            self.base_acq = None

    def _compute_acq(self, x):
        # 合わせよう
        if self.is_delta:
            sampled_ts, _ = self.model_g.predict(x)
        else:
            sampled_ts = self.model_g.posterior_samples(x, self.n_sampling_points)
        print(x.shape)
        return np.sum([self.base_acq.acquisition_function(
            np.hstack((x, [[t + self.tau_n, t] for _ in range(x.shape[0])]))) for t in sampled_ts], axis=0) / len(
            sampled_ts)

    def _compute_acq_withGradients(self, x):
        # 合わせよう
        if self.is_delta:
            sampled_ts = self.model_g.predict(x)
        else:
            sampled_ts = self.model_g.posterior_samples(x, self.n_sampling_points)
        f_acqus = []
        df_acqus = []
        print(x.shape)
        for t in sampled_ts:
            f_acqu, df_acqu = self.base_acq.acquisition_function_withGradients(
                np.hstack((x, [[t + self.tau_n, t] for _ in range(x.shape[0])])))
            f_acqus.append(f_acqu)
            df_acqus.append(df_acqu)
        return np.sum(f_acqus) / len(f_acqus), np.sum(df_acqus) / len(df_acqus)


class AcquisitionContinuousTimeVarying_MCMC(AcquisitionContinuousTimeVarying):

    def __init__(
            self,
            model,
            space,
            optimizer=None,
            cost_withGradients=None,
            **kwargs
    ):
        super(AcquisitionContinuousTimeVarying_MCMC, self).__init__(model, space, optimizer, **kwargs)
        self.base_acquisition_type = kwargs.get('base_acquisition_type', 'LCBforCTV_MCMC')
        self.model_g = kwargs.get('model_g', None)
        self.tau_n = kwargs.get('tau_n', None)
        self.n_sampling_points = kwargs.get('n_sampling_points', 30)
        self.is_delta = kwargs.get('is_delta', False)
        if self.model_g is None or self.tau_n is None:
            raise ValueError('model_g and tau_n should not be None.')
        self.optimizer = optimizer

        acquisition_jitter = kwargs.get('acquisition_jitter', 0.01)
        acquisition_weight = kwargs.get('acquisition_weight', 2)

        if self.base_acquisition_type == 'LCBforCTV_MCMC':
            self.base_acq = AcquisitionLCBforCTV_MCMC(
                self.model,
                self.space,
                self.optimizer,
                None,
                **kwargs
            )
        else:
            raise ValueError("The base_acquisitin_type should not be {}.".format(self.base_acquisition_type),
                             "It should be one of [LCBforCTV].")

    def _compute_acq(self, x):
        if self.is_delta:
            sampled_ts, _ = self.model_g.predict(x)
            sampled_ts = sampled_ts[:, :, None]
        else:
            sampled_ts = self.model_g.posterior_samples(x, self.n_sampling_points)
        sampled_ts = sampled_ts.reshape((sampled_ts.shape[0], sampled_ts.shape[2]))
        facq_s = np.zeros((sampled_ts.shape[1], x.shape[0], 1))
        for i in range(sampled_ts.shape[1]):
            sampled_t = sampled_ts[:, i]
            sampled_t = sampled_t.reshape((len(sampled_t), 1))
            time_list = np.hstack((sampled_t + self.tau_n, sampled_t))
            stacked_x = np.hstack((x, time_list))
            hoge = self.base_acq.acquisition_function(stacked_x)
            facq_s[i, :, :] = hoge
        summed_facq = np.sum(facq_s, axis=0)
        ret = summed_facq / len(sampled_ts)
        return ret

    def _compute_acq_withGradients(self, x):
        # 上に合わせよう
        if self.is_delta:
            sampled_ts = self.model_g.predict(x)
            sampled_ts = sampled_ts[:, :, None]
        else:
            sampled_ts = self.model_g.posterior_samples(x, self.n_sampling_points)
        sampled_ts = sampled_ts.reshape((sampled_ts.shape[0], sampled_ts.shape[2]))
        f_acqus = []
        df_acqus = []
        for t in sampled_ts:
            f_acqu, df_acqu = self.base_acq.acquisition_function_withGradients(
                np.hstack((x, [[t + self.tau_n, t] for _ in range(x.shape[0])])))
            f_acqus.append(f_acqu)
            df_acqus.append(df_acqu)
        return np.sum(f_acqus) / len(f_acqus), np.sum(df_acqus) / len(df_acqus)
