"""
Risk Functions
https://riskfolio-lib.readthedocs.io/en/latest/_modules/RiskFunctions.html

Morningstar - Custom Calculation Data Points
https://morningstardirect.morningstar.com/clientcomm/customcalculations.pdf
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.api as sm
import warnings


def Alpha(r_p: float, r_m: float, beta: float, r_f: float = 0.0) -> float:
    r"""
    Calculate the Alpha of a returns series.

    Parameters
    ----------
    r_p : float
        Portfolio expected return (mean).
        Dependant variable.
    r_m : float
        Benchmark expected return (mean).
        Independant variable.
    beta : float
        Beta of the portfolio.
    r_f : float, optional
        Risk free rate. The default is 0.0.

    .. math:: R_p-R_f=\alpha+\beta \left (R_m-R_f \right )+\varepsilon

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Alpha of a returns series.
    """

    y = r_p - r_f
    x = r_m - r_f

    value = y - beta * x

    return value


def Alpha_regression(X: np.ndarray, Y: np.ndarray, r_f: float = 0.0) -> float:
    r"""
    Calculate the Alpha of a returns series.

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
        Market index returns.
    Y : 1d-array
        Returns series, must have Tx1 size.
        Asset returns.
    r_f : float
        Risk free rate.

    .. math:: R_{i,t}-R_{f}=\alpha _{i}+\beta _{i}\,(R_{M,t}-R_{f})+\varepsilon _{i,t}

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Alpha of a returns series.
    """

    a = np.array(X, ndmin=2)
    b = np.array(Y, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if b.shape[0] == 1 and b.shape[1] > 1:
        b = b.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if b.shape[0] > 1 and b.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # The alpha coefficient is a parameter in the single-index model (SIM).
    # It is the intercept of the security characteristic line (SCL), that is,
    # the coefficient of the constant in a market model regression.
    X = sm.add_constant(a - r_f)
    model = sm.OLS(b - r_f, X).fit()
    value = model.params[0]

    return value


def Beta(X: np.ndarray, Y: np.ndarray) -> float:
    r"""
    Calculate the Beta of a returns series.

    .. math:: \beta(Y, X) = \frac{\text{Cov}(X,Y)}{\text{Var}(Y)}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
        Market index returns.
    Y : 1d-array
        Returns series, must have Tx1 size.
        Asset returns.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Beta of a returns series.
    """

    a = np.array(X, ndmin=2)
    b = np.array(Y, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if b.shape[0] == 1 and b.shape[1] > 1:
        b = b.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if b.shape[0] > 1 and b.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    T, N = a.shape
    mu_a = np.mean(a, axis=0).reshape(1, -1)
    mu_a = np.repeat(mu_a, T, axis=0)
    mu_b = np.mean(b, axis=0).reshape(1, -1)
    mu_b = np.repeat(mu_b, T, axis=0)
    cov = np.sum((b - mu_b) * (a - mu_a)) / T
    var = np.sum((a - mu_a) ** 2) / T
    value = cov / var

    return value


def Beta_regression(X: np.ndarray, Y: np.ndarray) -> float:
    r"""
    Calculate the Beta of a returns series using linear regression.

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
        Independent variable (for example, the market index).
    Y : 1d-array
        Returns series, must have Tx1 size.
        Dependent variable (for example, the asset returns).

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Beta of a returns series.
    """

    a = np.array(X, ndmin=2)
    b = np.array(Y, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if b.shape[0] == 1 and b.shape[1] > 1:
        b = b.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if b.shape[0] > 1 and b.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # Beta is measured as the slope of the regression of the excess return on the fund as the
    # dependent variable and the excess return on the risk-free rate as the independent variable.
    X = sm.add_constant(a)
    model = sm.OLS(b, X).fit()
    value = model.params[1]

    return value


def Kurtosis(X: np.ndarray) -> float:
    r"""
    Calculate the Square Root Kurtosis of a returns series.

    kurtosis for normal distribution is equal to 3.

    For a distribution having kurtosis < 3: It is called playkurtic.

    For a distribution having kurtosis > 3, It is called leptokurtic and it signifies that it tries to produce more outliers rather than the normal distribution.

    .. math::
        \text{Kurt}(X) = \frac{\mathbb{E}\left[(X - \mathbb{E}(X))^{4}\right]}{\sqrt{\mathbb{E}\left[(X-\mathbb{E}(X))^2\right]}^4}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Square Root Kurtosis of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    T, N = a.shape
    mu = np.mean(a, axis=0).reshape(1, -1)
    mu = np.repeat(mu, T, axis=0)
    n = np.sum(np.power(a - mu, 4)) / T
    # std = np.sqrt(np.sum(np.power(a - mu, 2)) / T)
    # d = np.power(std, 4)
    d = np.power(np.sum(np.power(a - mu, 2)) / T, 2)
    value = n / d

    return value


def Skewness(X: np.ndarray) -> float:
    r"""
    Calculate the Skewness of a returns series.

    Distribution on the basis of skewness value:
        Skewness = 0: Then normally distributed.

        Skewness > 0: Then more weight in the right tail of the distribution.

        Skewness < 0: Then more weight in the left tail of the distribution.

    .. math::
        \text{Skew}(X) = \frac{1}{T}\sum_{t=1}^{T}
        \left ( \frac{X_{t} - \mathbb{E}(X_{t})}{\sigma(X_{t})} \right )^{3}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Skewness of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    T, N = a.shape
    mu = np.mean(a, axis=0).reshape(1, -1)
    mu = np.repeat(mu, T, axis=0)
    sigma = np.std(a, axis=0).reshape(1, -1)
    sigma = np.repeat(sigma, T, axis=0)
    value = (a - mu) / sigma
    value = np.sum(np.power(value, 3)) / T
    value = value.item()

    return value


def VaR_Hist(X: np.ndarray, alpha: float = 0.05) -> float:
    r"""
    Calculate the Value at Risk (VaR) of a returns series.

    .. math::
        \text{VaR}_{\alpha}(X) = -\inf_{t \in (0,T)} \left \{ X_{t} \in
        \mathbb{R}: F_{X}(X_{t})>\alpha \right \}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    alpha : float, optional
        Significance level of VaR. The default is 0.05.
    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        VaR of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    sorted_a = np.sort(a, axis=0)
    index = int(np.ceil(alpha * len(sorted_a)) - 1)
    value = -sorted_a[index]
    value = np.array(value).item()

    return value


def CVaR_Hist(X: np.ndarray, alpha: float = 0.05) -> float:
    r"""
    Calculate the Conditional Value at Risk (CVaR) of a returns series.

    .. math::
        \text{CVaR}_{\alpha}(X) = \text{VaR}_{\alpha}(X) +
        \frac{1}{\alpha T} \sum_{t=1}^{T} \max(-X_{t} -
        \text{VaR}_{\alpha}(X), 0)

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    alpha : float, optional
        Significance level of CVaR. The default is 0.05.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        CVaR of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    sorted_a = np.sort(a, axis=0)
    index = int(np.ceil(alpha * len(sorted_a)) - 1)
    sum_var = 0
    for i in range(0, index + 1):
        sum_var = sum_var + sorted_a[i] - sorted_a[index]

    value = -sorted_a[index] - sum_var / (alpha * len(sorted_a))
    value = np.array(value).item()

    return value


def EVaR_Hist(X: np.ndarray, alpha: float = 0.05, solver: str = "CLARABEL") -> tuple:
    r"""
    Calculate the Entropic Value at Risk (EVaR) of a returns series.

    .. math::
        \text{EVaR}_{\alpha}(X) = \inf_{z>0} \left \{ z
        \ln \left (\frac{M_X(z^{-1})}{\alpha} \right ) \right \}

    Where:

    :math:`M_X(t)` is the moment generating function of X.

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    alpha : float, optional
        Significance level of EVaR. The default is 0.05.
    solver: str, optional
        Solver available for CVXPY that supports exponential cone programming.
        Used to calculate EVaR and EDaR. The default value is 'CLARABEL'.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    (value, z) : tuple
        EVaR of a returns series and value of z that minimize EVaR.

    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    T, N = a.shape

    # Primal Formulation
    t = cp.Variable((1, 1))
    z = cp.Variable((1, 1), nonneg=True)
    ui = cp.Variable((T, 1))
    ones = np.ones((T, 1))

    constraints = [
        cp.sum(ui) <= z,
        cp.constraints.ExpCone(-a * 1000 - t * 1000, ones @ z * 1000, ui * 1000),
    ]

    risk = t + z * np.log(1 / (alpha * T))
    objective = cp.Minimize(risk * 1000)
    prob = cp.Problem(objective, constraints)

    try:
        if solver in ["CLARABEL", "MOSEK", "SCS"]:
            prob.solve(solver=solver)
        else:
            prob.solve()
    except:
        pass

    if risk.value is None:
        value = None
    else:
        value = risk.value.item()
        t = z.value

    if value is None:
        warnings.filterwarnings("ignore")

        bnd = Bounds([-1e-24], [np.inf])
        result = minimize(
            _Entropic_RM, [1], args=(X, alpha), method="SLSQP", bounds=bnd, tol=1e-12
        )
        t = result.x
        t = t.item()
        value = _Entropic_RM(t, X, alpha)

    return (value, t)


def r_squared(X: np.ndarray, Y: np.ndarray) -> float:
    r"""
    Calculate the R-squared of returns series.

    .. math::
        R^{2} = 1 - \frac{\sum_{t=1}^{T}(Y_{t}-\hat{Y}_{t})^{2}}{\sum_{t=1}^{T}(Y_{t}-\bar{Y})^{2}}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
        Market index returns.
    Y : 1d-array
        Returns series, must have Tx1 size.
        Asset returns.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        R-squared of a returns series.
    """

    a = np.array(X, ndmin=2)
    b = np.array(Y, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if b.shape[0] == 1 and b.shape[1] > 1:
        b = b.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if b.shape[0] > 1 and b.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # The coefficient of determination, R-squared, is a measure of the proportion of the
    # variance in the dependent variable that is predictable from the independent variable(s).
    X = sm.add_constant(a)
    model = sm.OLS(b, X).fit()
    value = model.rsquared

    return value


def tracking_error(X: np.ndarray, Y: np.ndarray) -> float:
    r"""
    Calculate the Tracking Error of a returns series.

    .. math::
        TE = \sqrt{\sum_{t=1}^{T}(Y_{t}-\hat{Y}_{t})^{2}/(T-1)}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
        Market index returns.
    Y : 1d-array
        Returns series, must have Tx1 size.
        Asset returns.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Tracking Error of a returns series.
    """

    a = np.array(X, ndmin=2)
    b = np.array(Y, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if b.shape[0] == 1 and b.shape[1] > 1:
        b = b.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")
    if b.shape[0] > 1 and b.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # The tracking error is the standard deviation of the difference between the portfolio
    # and benchmark returns.
    T = len(b)
    value = np.sqrt(np.sum(b - a) ** 2 / (T - 1))

    return value


def information_ratio(r_p: float, r_m: float, te_pm: float) -> float:
    r"""
    Calculate the Information Ratio of a returns series.

    .. math::
        IR = \frac{R_p-R_m}{TrackingError(p,m)}

    Parameters
    ----------
    r_p : float
        Portfolio expected return (mean).
    r_m : float
        Benchmark expected return (mean).
    te_pm : float
        Tracking error between portfolio and benchmark.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Information Ratio of a returns series.
    """

    # The information ratio is a measure of portfolio returns beyond the returns of a benchmark,
    # usually an index, compared to the volatility of those returns.
    value = (r_p - r_m) / te_pm

    return value


def treynor_ratio(r_p: float, beta: float, r_f: float = 0.0) -> float:
    r"""
    Calculate the Treynor Ratio of a returns series.
    Useful to compare funds with different risk levels.

    .. math::
        TR = \frac{R_p-R_f}{\beta}

    Parameters
    ----------
    r_p : float
        Portfolio expected return (mean).
    beta : float
        Beta of the portfolio.
    r_f : float, optional
        Risk free rate. The default is 0.0.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Treynor Ratio of a returns series.
    """

    value = (r_p - r_f) / beta

    return value


def max_drawdown(X: np.ndarray) -> float:
    r"""
    Calculate the Maximum Drawdown of a prices series.

    .. math::
        MDD = \max_{t \in (0,T)} \left \{ \frac{X_{t}-\max_{t \in (0,T)}X_{t}}{\max_{t \in (0,T)}X_{t}} \right \}

    Parameters
    ----------
    X : 1d-array
        Prices series, must have Tx1 size.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Maximum Drawdown of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    a = pd.Series(a.flatten())
    value = np.max(np.abs((a - a.cummax()) / a.cummax()))

    return value


def average_drawdown(X: np.ndarray, min_dd_length: int = 5) -> tuple:
    r"""
    Calculate the Average Drawdown of a price series.

    Parameters
    ----------
    X : DataFrame
        Price series, must have Tx1 size.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    df : DataFrame
        DataFrame with the following columns:
        - Close: Price series
        - tuw: Time Under Water
        - dd: Drawdown
        - tuw_idx: Time Under Water index
        - dd_length: Drawdown length
    num_dd : int
        Number of drawdowns
    avg_dd : float
        Average drawdown
    avg_dd_length : float
        Average drawdown length
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # Create a DataFrame with price serie
    column = "Close"
    df = pd.DataFrame(X, columns=[column])

    df["tuw"] = df[column].cummax()  # Time Under Water
    df["dd"] = df[column] / df["tuw"] - 1
    df["tuw_idx"] = (
        df[column]
        .expanding(min_periods=1)
        .apply(lambda x: x.argmax())
        .fillna(0)
        .astype(int)
    )
    df["dd_length"] = (df["tuw_idx"] - df["tuw_idx"].shift(1) - 1).fillna(0)
    df["dd_length"] = df["dd_length"][df["dd_length"] > min_dd_length]
    df["dd_length"].fillna(0, inplace=True)
    dd_end_idx = df["tuw_idx"].loc[df["dd_length"] != 0]
    temp_dd_days = df["dd_length"].loc[df["dd_length"] != 0]
    dd_start_idx = dd_end_idx - temp_dd_days
    temp_dd = [
        np.min(df["dd"].loc[dd_start_idx.iloc[i] : dd_end_idx.iloc[i]])
        for i in range(len(dd_end_idx))
    ]
    num_dd = len(temp_dd)
    avg_dd = np.average(temp_dd)
    avg_dd_length = (df["dd_length"][df["dd_length"] > 0]).mean()

    return df, num_dd, avg_dd, avg_dd_length


def annualized_return(X: np.ndarray, num_periods: int = 252) -> float:
    r"""
    Calculate the Annualized Return of a prices series.
    Ref.: https://www.allquant.co/post/magic-of-log-returns-practical-part-2#:~:text=Using%20Log%20Returns%20–%20We%20multiply,if%20monthly%2C%20then%20by%2012.

    .. math::
        AR = \left (e^{\frac{252}{T} \sum_{t=1}^{T} logrets} \right )- 1

    Parameters
    ----------
    X : 1d-array
        Prices series, must have Tx1 size.
    num_periods : int, optional
        Number of periods. The default is 252 for daily prices.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Annualized Return of a prices series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # Compute log returns
    a = pd.Series(a.flatten())
    log_returns = np.log(a).diff().dropna()

    value = np.exp(num_periods * np.mean(log_returns)) - 1

    return value


def annualized_volatility(X: np.ndarray, num_periods: int = 252) -> float:
    r"""
    Calculate the Annualized Volatility of a returns series.

    .. math::
        AV = \sqrt{\frac{252}{T} \sum_{t=1}^{T} logrets}

    Parameters
    ----------
    X : 1d-array
        Returns series, must have Tx1 size.
    num_periods : int, optional
        Number of periods. The default is 252 for daily returns.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Annualized Volatility of a returns series.
    """

    a = np.array(X, ndmin=2)
    if a.shape[0] == 1 and a.shape[1] > 1:
        a = a.T
    if a.shape[0] > 1 and a.shape[1] > 1:
        raise ValueError("returns must have Tx1 size")

    # value = np.sqrt(num_periods * np.var(a))
    value = np.sqrt(num_periods) * np.std(a)

    return value


def calmar_ratio(r_p: float, max_dd: float, r_f: float = 0.0) -> float:
    r"""
    Calculate the Calmar Ratio.

    .. math::
        CR = \frac{R_p-R_f}{MaxDrawdown(p)}

    Parameters
    ----------
    r_p : float
        Annualized portfolio return.
    max_dd : float
        Maximum drawdown.
    r_f : float, optional
        Annualized risk free rate. The default is 0.0.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    value : float
        Calmar Ratio.
    """

    value = (r_p - r_f) / max_dd

    return value
