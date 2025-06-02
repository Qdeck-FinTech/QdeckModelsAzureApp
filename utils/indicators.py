
import numba
import numpy as np
from scipy.stats import norm


EPS = np.finfo(np.float32).eps


@numba.njit
def rsi(prices,window):
    rets = np.log(prices[:window]) - np.log(prices[1:window+1])
    up = np.sum(rets*(rets>0),axis=0)/(np.sum(rets>0,axis=0) + EPS)
    down = np.sum(rets*(rets<0),axis=0)/(np.sum(rets<0,axis=0) + EPS)
    return 100 - 100/(1+np.abs( up/down ))


@numba.njit
def usrsi(prices,window):
    rets = np.log(prices[:window]) - np.log(prices[1:window+1])
    up = np.sum(rets*(rets>0),axis=0)
    down = np.sum(rets*(rets<0),axis=0) + EPS
    return 100 - 100/(1+np.abs( up/down ))


@numba.njit
def ema_r(prices,span):
    rets = np.log(prices[:-1]) - np.log(prices[1:])
    return ema(rets,span)


@numba.njit
def ema(prices,span):
    alpha = 2/(span+1)
    weights = np.cumprod(np.concatenate((np.ones(1),np.repeat(1-alpha,len(prices)-1))))
    return weights @ prices / np.sum(weights)


@numba.njit
def logret(prices,window):
    return np.log(prices[0]) - np.log(prices[window-1])


@numba.njit
def ret(prices,window):
    return (prices[0] - prices[window-1])/prices[0]


# @numba.njit
def pctoh(prices,window):
    high = np.amax(prices[:window],axis=0)
    return np.log(high) - np.log(prices[0])


# @numba.njit
def pctol(prices,window):
    low = np.amin(prices[:window],axis=0)
    return np.log(low) - np.log(prices[0])


@numba.njit
def snr_st(prices,state,limit,window):
    if state.size == 0:
        if np.sum(prices !=0 ) < window+1:
            return state
        s = 1
    else:
        s = state[-1]

    rets = np.log(prices[:window]) - np.log(prices[1:window+1])
    snr = np.sum(rets) / (np.sum(np.abs(rets)) + EPS)
    
    if s == 1 and snr + limit < 0:
        return np.append(state,-1)
    elif s == -1 and snr - limit > 0:
        return np.append(state,1)
    return np.append(state,s)


def bfactor(prices,window):
    rets = np.log(prices[:window]) - np.log(prices[1:window+1])
    mu = np.mean(rets,axis=0)
    sigma = np.std(rets,axis=0)
    return np.e**np.mean(norm.logpdf(rets,mu,sigma) - norm.logpdf(rets,0.0,sigma),axis=0)