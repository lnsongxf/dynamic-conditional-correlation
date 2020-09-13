def optimize(file, actif_1, actif_2, period_1, period_2):
    data = pd.read_excel(file)
    df = data[['Date', actif_1, actif_2]]
    year1, month1, day1 = period_1.split('-')
    year1 = int(year1)
    month1 = int(month1)
    day1 = int(day1)
    year2, month2, day2 = period_2.split('-')
    year2 = int(year2)
    month2 = int(month2)
    day2 = int(day2)
    df = df[datetime.datetime(year1, month1, day1, 0, 0, 0) <= df.Date]
    df = df[df.Date <= datetime.datetime(year2, month2, day2, 0, 0, 0)]

    returns = df.mean().to_dict()
    ecart_type = df.std().to_dict()
    R_f = 0.001
    rho = df.corr().to_dict()[actif_1][actif_2]
    beta = 1
    # optimize 
    from scipy.optimize import minimize
    from scipy.optimize import LinearConstraint
    #Create initial point.
    x0 = [0.3 ,0.8]
    #Create function to be minimized
    def objective(x):
        alpha_1 = x[0]
        alpha_2 = x[1]
        max_ = alpha_1 * returns[actif_1] + alpha_2 * returns[actif_2] - R_f
        min_ = beta * (alpha_1 ** 2) * (ecart_type[actif_1] ** 2) + (alpha_2 ** 2) * (ecart_type[actif_2] ** 2) + 2 * alpha_1 * alpha_2 * ecart_type[actif_1] * ecart_type[actif_2] * rho
        return max_ - min_

    A = np.array([1, 1]).reshape(1, 2)
    lbnd = upbnd = 1
    lin_cons = LinearConstraint(A,lbnd,upbnd)
    sol = minimize(objective,x0,constraints=lin_cons)['x']

    # 
    #return sol['x'], df
    pivoted = df.set_index('Date')
    pivoted.rename_axis(columns="ticker")
    cov_matrix = pivoted.apply(lambda x: np.log(1+x)).cov()
    e_r = pivoted.resample('Y').last().pct_change().mean()
    sd = pivoted.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    assets = pd.concat([e_r, sd], axis=1)
    assets.columns = ['Returns', 'Volatility']
    p_ret = []
    p_vol = []
    p_weights = []

    num_portfolios = 1
    for portfolio in range(num_portfolios):
        weights = sol
        p_weights.append(weights)
        returns = np.dot(weights, e_r)
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd*np.sqrt(250)
        p_vol.append(ann_sd)

    data = {'Returns':p_ret, 'Volatility':p_vol}
    portfolios  = pd.DataFrame(data)
    portfolios.index = ['portfolio1']
    op_space = pd.concat([portfolios, assets])
    return op_space
  