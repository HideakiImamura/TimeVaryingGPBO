import optuna
import numpy as np
import time
import argparse
import os


def quantize(a, quantization):
    return np.argmin(np.abs(np.linspace(0., 1., quantization) - a))


def fun_bogunovic(a, b, quantization):
    global t
    t += 1
    i = quantize(a, quantization)
    j = quantize(b, quantization)
    place = i * quantization + j
    time.sleep(1)
    period = data.shape[0]
    best_values.append(best_data[t % period])
    return data[t][place]


def fun_original(a, b, quantization):
    global t
    t += 1
    i = quantize(a)
    j = quantize(b)
    place = i * quantization + j
    if place < quantization ** 2 // 2:
        time.sleep(1)
    else:
        time.sleep(10)
    period = data.shape[0]
    best_values.append(best_data[t % period])
    return data[t][place]


def fun_tv_and_time_dep(a, b, quantization):
    global t
    i = quantize(a, quantization)
    j = quantize(b, quantization)
    print(i, j)
    place = i * quantization + j
    if place < quantization ** 2 // 2:
        time.sleep(1)
        t += 1
    else:
        time.sleep(10)
        t += 10
    period = data.shape[0]
    best_values.append(best_data[t % period])
    return data[t % period][place]


def objective_bogunovic(trial, quantization):
    a, b = trial.suggest_joint('ab',
                               [['uniform', 'a', 0., 1.],
                                ['uniform', 'b', 0., 1.]])
    return fun_bogunovic(a, b, quantization)


def objective_original(trial, quantization):
    a, b = trial.suggest_joint('ab',
                               [['uniform', 'a', 0., 1.],
                                ['uniform', 'b', 0., 1.]])
    return fun_original(a, b, quantization)


def objective_tv_and_time_dep(trial, quantization):
    a, b = trial.suggest_joint('ab',
                               [['uniform', 'a', 0., 1.],
                                ['uniform', 'b', 0., 1.]])
    return fun_bogunovic(a, b, quantization)


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed=seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--function-type', default='rbf')
    parser.add_argument('--objective-type', default='bogunovic')
    parser.add_argument('--sampler-type', default='ctv')
    parser.add_argument('--target-dir', default='./results/')
    parser.add_argument('--epsilon', default=-1., type=float)
    parser.add_argument('--n-sampling-points', default=30, type=int)
    parser.add_argument('--space-kernel', default='Matern52')
    parser.add_argument('--time-kernel', default='Bogunovic')
    parser.add_argument('--g-kernel', default='Matern52')
    parser.add_argument('--kernel-selection-experiment', default=False, type=bool)
    args = parser.parse_args()

    data = np.loadtxt('functions/true_bogunovic_' + args.function_type + '.csv', delimiter=',')
    best_data = np.max(data, axis=1)

    t = -1
    quantization = int(np.sqrt(data.shape[1]))

    n_trials = int(data.shape[0])

    if args.objective_type == 'bogunovic':
        obj = objective_bogunovic
    elif args.objective_type == 'original':
        obj = objective_original
    elif args.objective_type == 'tv_and_time_dep':
        obj = objective_tv_and_time_dep
    else:
        raise ValueError("Objective type = {bogunovic, original, tv_and_time_dep}.")

    if args.epsilon < 0.:
        epsilon = 0.01
    else:
        epsilon = args.epsilon

    n_sampling_points = args.n_sampling_points

    if args.sampler_type == 'normal':
        sampler = optuna.samplers.GPSampler(
            kernel=args.space_kernel
        )
    elif args.sampler_type == 'dtv':
        sampler = optuna.samplers.DTVGPSampler(
            space_kernel=args.space_kernel,
            time_kernel=args.time_kernel,
            epsilon=epsilon)
    elif args.sampler_type == 'ctv':
        sampler = optuna.samplers.CTVGPSampler(
            space_kernel=args.space_kernel,
            time_kernel=args.time_kernel,
            g_kernel=args.g_kernel,
            n_sampling_points=n_sampling_points,
            epsilon=epsilon)
    elif args.sampler_type == 'ctv_delta':
        sampler = optuna.samplers.CTVGPSampler(
            is_delta=True,
            space_kernel=args.space_kernel,
            time_kernel=args.time_kernel,
            g_kernel=args.g_kernel,
            epsilon=epsilon
        )
    else:
        raise ValueError("Sampler type = {normal, dtv, ctv}.")

    best_values = []

    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: - obj(t, quantization), n_trials=n_trials)

    trial_values = [- trial.value for trial in study.trials]

    target_dir = args.target_dir
    if args.epsilon >= 0.:
        file_name = 'syn_' + args.objective_type + '_' + args.function_type + '_' + args.sampler_type + str(args.epsilon) + '.csv'
    elif args.n_sampling_points is not 30:
        file_name = 'syn_' + args.objective_type + '_' + args.function_type + '_' + args.sampler_type + str(args.n_sampling_points) + '.csv'
    elif args.kernel_selection_experiment:
        file_name = 'syn_' + args.objective_type + \
                    '_' + args.function_type + \
                    '_' + args.sampler_type + \
                    '_Space_' + args.space_kernel + \
                    '_Time_' + args.time_kernel + \
                    '_G_' + args.g_kernel + '.csv'
    else:
        file_name = 'syn_' + args.objective_type + '_' + args.function_type + '_' + args.sampler_type + '.csv'


    np.savetxt(
        target_dir + file_name,
        trial_values,
        delimiter=',')
    np.savetxt(
        target_dir + 'best_' + file_name,
        best_values,
        delimiter=','
    )
