import matplotlib.pyplot as mpl
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import random


def get_correlation_distance(corr):
    dist = ((1 - corr) / 2.) ** .5
    return dist


def get_quasi_diagram(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index;
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def get_ivp(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_variance(cov, cItems):
    cov_ = cov.loc[cItems, cItems]
    w_ = get_ivp(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar


def get_rec_bi_part(cov, sortIx):
    # compute HRP allocation
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i + 1]
            cVar0 = get_cluster_variance(cov, cItems0)
            cVar1 = get_cluster_variance(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] += alpha
            w[cItems1] += 1 - alpha
    return w


def get_hrp(cov, corr):
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = get_correlation_distance(corr)
    link = sch.linkage(dist, 'single')
    sortIx = get_quasi_diagram(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = get_rec_bi_part(cov, sortIx)
    return hrp.sort_index()


def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F):
    x = np.random.normal(mu0, sigma0, size=(nObs, size0))
    # create correlation between variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # add common two shocks
    point_cs = np.random.randint(sLength, nObs - 1, size=2)
    x[np.ix_(point_cs, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    # add specific random shock
    point_rs = np.random.randint(sLength, nObs - 1, size=2)
    x[point_rs, cols[-1]] = np.array([-.5, 2])
    return x, cols, ((point_cs, [cols[0], size0]), (point_rs, cols[-1]))


def hrpMC(numIters=1e4, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2, sigma1F=.25, sLength=260, rebal=22):
    # Monte Carlo experiment on HRP
    methods = [get_hrp]
    stats, numIter = {i.__name__: pd.Series() for i in methods}, 0

    pointers = range(sLength, nObs, rebal)

    weight = {i.__name__: pd.DataFrame(index=range(sLength, nObs, rebal),
                                       columns=[str(j) for j in range(size0 * 2)]) for i in methods}

    while numIter < numIters:
        x, cols, shock = generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F)
        print(numIter, shock)
        pt0 = mpl.plot(x)
        fig0 = pt0[0].get_figure()
        fig0.savefig("price.shock.{0}.png".format(numIter))
        mpl.close()
        r = {i.__name__: pd.Series() for i in methods}
        # compute portfolios in sample
        for pointer in pointers:
            x_ = x[pointer - sLength: pointer]
            cov_, corr_ = np.cov(x_, rowvar=0), np.corrcoef(x_, rowvar=0)
            # compute the performance out of sample
            x_ = x[pointer: pointer + rebal]
            for func in methods:
                w_ = func(cov=cov_, corr=corr_)
                r_ = pd.Series(np.dot(x_, w_))
                weight[func.__name__].loc[pointer] = w_.values
                r[func.__name__] = r[func.__name__].append(r_)
        for func in methods:
            r_ = r[func.__name__].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[func.__name__].loc[numIter] = p_.iloc[-1] - 1.0
            pt = weight[func.__name__].plot(title="{0} {1}".format(numIter, shock), grid=True, figsize=(10, 6))
            fig = pt.get_figure()
            fig.savefig("{0}.weight.{1}.png".format(func.__name__, numIter))
        numIter += 1
    stats = pd.DataFrame.from_dict(stats, orient='columns')
    stats.to_csv('stats.csv')
    df0, df1 = stats.std(), stats.var()
    print(pd.concat([df0, df1, df1 / df1['get_hrp'] - 1.0], axis=1))
    return


hrpMC(numIters=2, rebal=22, nObs=225 * 10, size0=3, size1=3)