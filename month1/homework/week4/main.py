import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

random_var_params = {
    'a': 0.,
    'b': 10.,
    'c': .8,
}

random_var = sts.triang(
    random_var_params['c'],
    loc=random_var_params['a'],
    scale=random_var_params['b'] - random_var_params['a']
)

x_domain = np.linspace(
    random_var_params['a'] - 1,
    random_var_params['b'] + 1,
    1000
)


def first_task():
    sample = random_var.rvs(1000)
    theoretical_pdf = random_var.pdf(x_domain)

    plt.plot(x_domain, theoretical_pdf, label='theoretical pdf', color='blue')
    plt.hist(sample, normed=True, label='histogram of sample', color='orange', edgecolor='black', linewidth=0.5)
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.legend()
    plt.show()


def build_approx_ex(sample_size):
    sample_list = []
    for i in xrange(0, 1000):
        sample = random_var.rvs(sample_size)
        sample_list.append(sample)
    approx_ex_list = []
    for i in xrange(0, len(sample_list)):
        approx_ex = np.mean(sample_list[i])
        approx_ex_list.append(approx_ex)
    plt.title('sample size ' + str(sample_size))
    plt.hist(approx_ex_list, normed=True, label='histogram of ex',
             color='orange',
             edgecolor='black',
             linewidth=0.5)

    ex_mu = random_var.mean()
    ex_sigma = np.sqrt(random_var.var() / float(sample_size))
    ex_random_var = sts.norm(loc=ex_mu, scale=ex_sigma)
    ex_pdf = ex_random_var.pdf(x_domain)
    plt.plot(x_domain, ex_pdf, label='EX approximated norm', color='blue')
    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.legend()
    plt.show()


def second_task():
    build_approx_ex(5)


# first_task()
second_task()
