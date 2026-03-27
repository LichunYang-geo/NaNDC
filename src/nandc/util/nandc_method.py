import numpy as np
from xarray import DataArray
from pandas import DataFrame, Series, merge

from nandc_mt import tt_to_mij, get_full_MT_array

def normalize_by_percentile(mis, p=95.0):
    scale = np.percentile(mis, p)
    if scale <= 0:
        return mis  # 退化情况
    return mis / scale

def misfit_normalization(event, mis_pol, mis_amp, mis_spr, weight, normalization, p):

    mis_pol = mis_pol.reshape(-1, 1)
    mis_amp = mis_amp.reshape(-1, 1)
    mis_spr = mis_spr.reshape(-1, 1)
    if normalization:
        # Linear Normalization (max)
        norm_pol_misfit = mis_pol
        norm_amp_misfit = normalize_by_percentile(mis_amp, p) if weight[1] != 0 and len(event.amp) != 0 else mis_amp
        norm_spr_misfit = normalize_by_percentile(mis_spr, p) if weight[2] != 0 and len(event.spr) != 0 else mis_spr
        misfit_sum = weight[0]*norm_pol_misfit + weight[1]*norm_amp_misfit + weight[2]*norm_spr_misfit
        misfit_all = np.concatenate((misfit_sum.reshape(-1,1), 
                                     norm_pol_misfit.reshape(-1,1), 
                                     norm_amp_misfit.reshape(-1,1), 
                                     norm_spr_misfit.reshape(-1,1)), axis=1)
    else:
        misfit_sum = weight[0]*mis_pol + weight[1]*mis_amp + weight[2]*mis_spr
        misfit_all = np.concatenate((misfit_sum.reshape(-1,1), mis_pol, mis_amp, mis_spr), axis=1)

    return misfit_sum, misfit_all

def calc_optimal_misfit(syn_pol, syn_pamp, syn_spratio, obs_pol, obs_pamp, obs_spratio, 
                        tau1, tau2, tau3, lambd1, lambd2, lambd3, norm_list):
    '''
    Calculate misfit values for theoretical and observed data
    :param syn_pol: theoretical P-wave polarity
    :type syn_pol: np.ndarray
    :param syn_pamp: theoretical P-wave amplitude
    :type syn_pamp: np.ndarray
    :param syn_spratio: theoretical S/P amplitude ratio
    :type syn_spratio: np.ndarray
    :param obs_pol: observed P-wave polarity
    :type obs_pol: np.ndarray
    :param obs_pamp: observed P-wave amplitude
    :type obs_pamp: np.ndarray
    :param obs_spratio: observed S/P amplitude ratio
    :type obs_spratio: np.ndarray
    :param tau1: weighting for P-wave polarity in each station
    :type tau1: np.ndarray
    :param tau2: weighting for P-wave amplitude in each station
    :type tau2: np.ndarray
    :param tau3: weighting for S/P amplitude ratio in each station
    :type tau3: np.ndarray
    :return misfit: misfit values of each section
    :rtype misfit: np.ndarray
    '''

    if lambd1 != 0 and len(obs_pol) != 0 and len(tau1) != 0:
        if norm_list[0] == "L1":
            pol_mis = np.sum(tau1*0.5*np.abs(syn_pol-obs_pol), axis=1)/np.sum(tau1*np.abs(obs_pol))
        elif norm_list[0] == "L2":
            pol_mis = np.sum((tau1*0.5*(syn_pol-obs_pol))**2, axis=1)/np.sum((tau1*obs_pol)**2)
        else:
            raise ValueError("Invalid norm type. Use 'L1' or 'L2'.")
    else:
        pol_mis = np.zeros((np.size(syn_pol,0)))
    
    if lambd2 != 0 and len(obs_pamp) != 0 and len(tau2) != 0:
        if norm_list[1] == "L1":
            pamp_mis = np.sum(tau2*np.abs(syn_pamp-obs_pamp), axis=1)/np.sum(tau2*np.abs(obs_pamp))
        elif norm_list[1] == "L2":
            pamp_mis = np.sum((tau2*(syn_pamp-obs_pamp))**2, axis=1)/np.sum((tau2*obs_pamp)**2)
        else:
            raise ValueError("Invalid norm type. Use 'L1' or 'L2'.")
    else: 
        pamp_mis = np.zeros((np.size(syn_pamp,0)))

    if lambd3 != 0 and len(obs_spratio) != 0 and len(tau3) != 0:
        if norm_list[2] == "L1":
            ratio_mis = np.sum(tau3*np.abs(syn_spratio-obs_spratio), axis=1)/np.sum(tau3*np.abs(obs_spratio))
        elif norm_list[2] == "L2":
            ratio_mis = np.sum((tau3*(syn_spratio-obs_spratio))**2, axis=1)/np.sum((tau3*obs_spratio)**2)
        else:
            raise ValueError("Invalid norm type. Use 'L1' or 'L2'.")
    else:
        ratio_mis = np.zeros((np.size(syn_spratio,0)))

    return pol_mis, pamp_mis, ratio_mis

def get_best_fullspace(gs, asnumpy=True, mtype="NED"):
    '''
    extract the moment tensor with minimum misfit from full moment tensor space
    :param gs:
    :dtype gs: xarray.DataArray
    :param asnumpy: 
    :dtype asnumpy: bool
    '''
    # 保证的是“只取一个”，不是“全局最优解在数学上唯一” ！

    if not isinstance(gs, DataArray):
        raise ValueError("`gs` must be a `xarray.DataArray`")

    # 1. 把所有维度压缩成一个维度 'z'
    gstack = gs.stack(z=gs.dims)  # dims: ('z',)

    # 2. 找到全局最小值在 'z' 维度上的位置 (整数索引)
    z_min_idx = int(gstack.argmin(dim="z"))
    # 最小 misfit 数值
    min_val = float(gstack.isel(z=z_min_idx).values)

    # 3. 对应的坐标是 z 维度上的一个 MultiIndex 元素
    z_index = gstack.coords["z"].to_index()  # 这是一个 MultiIndex，names=原始各维度名
    mi = z_index[z_min_idx]                  # 比如一个 tuple: (rho_val, v_val, w_val, kappa_val, sigma_val, h_val)
    dim_names = z_index.names                # 比如 ('rho', 'v', 'w', 'kappa', 'sigma', 'h')

    # 4. 把这组坐标转成 DataFrame 的一行
    coord_df = DataFrame([mi], columns=dim_names)  # 1×6
    coord_df["misfit"] = min_val                      # 加上误差列 → 1×7

    # 为了和原代码保持一致，列顺序按你的习惯排一下
    expected_cols = ["rho", "v", "w", "kappa", "sigma", "h", "misfit"]
    cols = [c for c in expected_cols if c in coord_df.columns]
    tt_best_full = coord_df[cols]

    # 5. 转成矩张量
    mt_best_full = tt_to_mij(
        tt_best_full["rho"].to_numpy(),   tt_best_full["v"].to_numpy(),     tt_best_full["w"].to_numpy(),
        tt_best_full["kappa"].to_numpy(), tt_best_full["sigma"].to_numpy(), tt_best_full["h"].to_numpy(), mtype)

    # 6. 返回
    if asnumpy:
        return tt_best_full.to_numpy(), mt_best_full
    else:
        return tt_best_full, mt_best_full

def get_best_lunespace(gs, asnumpy=True, mtype="NED"):
    '''
    extract each source types' minimum fault oritation and its misfit from full moment tensor space
    :param gs: inversion DataArray
    :type gs: xarray.DataArray
    :param asnumpy: 
    '''
    # 保证的是“每个 (v,w) 只保留一个代表解”，不是“该 (v,w) 下最优断层取向唯一”

    # 1. 类型检查
    if not isinstance(gs, DataArray):
        raise ValueError("`gs` must be a `xarray.DataArray`")

    # 2. 需要沿这些维度寻找最小值（fault orientation）
    orient_dims = ("rho", "kappa", "sigma", "h")
    missing = [d for d in orient_dims if d not in gs.dims]
    if missing:
        raise ValueError(f"`gs` is missing orientation dims: {missing}")

    # 3. 每个 (v,w) 上的最小 misfit
    min_value = gs.min(dim=orient_dims)  # dims 通常为 ('v', 'w')

    # 4. 把 orientation 维度 stack 成一个新维度 'orient'
    gstack = gs.stack(orient=orient_dims)  # dims: ('orient', 'v', 'w')

    # 5. 对每个 (v,w) 沿 'orient' 找最小值的坐标标签
    #    即：每个 (v,w) 对应一个 MultiIndex(rho,kappa,sigma,h)
    orient_idx = gstack.idxmin(dim="orient")  # dims: ('v','w')

    # 6. 把 orient_idx 转成 Series，再把 MultiIndex 拆成 rho,kappa,sigma,h 四列
    orient_series = orient_idx.to_series()  # index: (v,w), value: MultiIndex
    orient_df = (
        orient_series
        .apply(lambda mi: Series(mi, index=orient_dims))  # 每个 mi 是一个 MultiIndex entry
        .reset_index()  # 把 index 中的 v,w 也变成列
    )
    # orient_df 列大致为: ['v', 'w', 'rho', 'kappa', 'sigma', 'h']

    # 7. 每个 (v,w) 对应的最小 misfit
    misfit_df = (
        min_value
        .to_series()         # index: (v,w)，value: misfit
        .rename("misfit")
        .reset_index()       # 列: ['v', 'w', 'misfit']
    )

    # 8. 按 (v,w) 合并，保证一一对应（one_to_one 避免意外多解）
    lune_dims = list(min_value.dims)  # 一般是 ['v', 'w']
    tt_best_lune = merge(
        orient_df,
        misfit_df,
        on=lune_dims,
        how="inner",
        validate="one_to_one",
    )

    # 9. 固定列顺序（和你原来的一致）
    tt_best_lune = tt_best_lune[["rho", "v", "w", "kappa", "sigma", "h", "misfit"]]

    # 10. 转成矩张量
    mt_best_lune = tt_to_mij(
        tt_best_lune["rho"].to_numpy(),   tt_best_lune["v"].to_numpy(),     tt_best_lune["w"].to_numpy(),
        tt_best_lune["kappa"].to_numpy(), tt_best_lune["sigma"].to_numpy(), tt_best_lune["h"].to_numpy(), mtype)

    # 11. 控制返回格式
    if asnumpy:
        return tt_best_lune.to_numpy(), mt_best_lune
    else:
        return tt_best_lune, mt_best_lune
    
def get_accepted_fullspace(gs, tolerance, asnumpy=True, mtype="NED"):
    '''
    Extract all accepted solutions in the full moment tensor space whose
    misfit is less than or equal to the given tolerance.

    Parameters
    ----------
    gs : xarray.DataArray
        Inversion misfit grid.
    tolerance : float
        Absolute misfit threshold.
    asnumpy : bool, default=True
        If True, return solution table as numpy array; otherwise return DataFrame.
    mtype : str, default="NED"
        Moment tensor coordinate convention passed to tt_to_mij.
    '''
    if not isinstance(gs, DataArray):
        raise ValueError("`gs` must be a `xarray.DataArray`")

    gstack = gs.stack(z=gs.dims)
    accepted = gstack.where(gstack <= tolerance, drop=True)

    expected_cols = ["rho", "v", "w", "kappa", "sigma", "h", "misfit"]

    if accepted.sizes.get("z", 0) == 0:
        tt_accept_full = DataFrame(columns=expected_cols)
        mt_accept_full = np.empty((0, 6))
        if asnumpy:
            return tt_accept_full.to_numpy(), mt_accept_full
        else:
            return tt_accept_full, mt_accept_full

    coord_df = accepted.coords["z"].to_index().to_frame(index=False)
    coord_df["misfit"] = accepted.values
    tt_accept_full = coord_df[expected_cols]

    mt_accept_full = tt_to_mij(
        tt_accept_full["rho"].to_numpy(),
        tt_accept_full["v"].to_numpy(),
        tt_accept_full["w"].to_numpy(),
        tt_accept_full["kappa"].to_numpy(),
        tt_accept_full["sigma"].to_numpy(),
        tt_accept_full["h"].to_numpy(),
        mtype,
    )

    if asnumpy:
        return tt_accept_full.to_numpy(), mt_accept_full
    else:
        return tt_accept_full, mt_accept_full

def get_accepted_lunespace(gs, tolerance, asnumpy=True, mtype="NED"):
    '''
    Extract accepted source types in lune space. For each (v, w), only the
    best fault orientation is retained first, then accepted solutions are
    selected by the given misfit tolerance.

    Parameters
    ----------
    gs : xarray.DataArray
        Inversion misfit grid.
    tolerance : float
        Absolute misfit threshold.
    asnumpy : bool, default=True
        If True, return solution table as numpy array; otherwise return DataFrame.
    mtype : str, default="NED"
        Moment tensor coordinate convention passed to tt_to_mij.
    '''
    if not isinstance(gs, DataArray):
        raise ValueError("`gs` must be a `xarray.DataArray`")

    tt_best_lune, _ = get_best_lunespace(gs, asnumpy=False, mtype=mtype)
    tt_accept_lune = tt_best_lune[tt_best_lune["misfit"] <= tolerance].reset_index(drop=True)

    if len(tt_accept_lune) == 0:
        mt_accept_lune = np.empty((0, 6))
        if asnumpy:
            return tt_accept_lune.to_numpy(), mt_accept_lune
        else:
            return tt_accept_lune, mt_accept_lune

    mt_accept_lune = tt_to_mij(
        tt_accept_lune["rho"].to_numpy(),
        tt_accept_lune["v"].to_numpy(),
        tt_accept_lune["w"].to_numpy(),
        tt_accept_lune["kappa"].to_numpy(),
        tt_accept_lune["sigma"].to_numpy(),
        tt_accept_lune["h"].to_numpy(),
        mtype,
    )

    if asnumpy:
        return tt_accept_lune.to_numpy(), mt_accept_lune
    else:
        return tt_accept_lune, mt_accept_lune

def get_best_from_result_table(result_table, asnumpy=True, mtype="NED"):
    result_table = np.asarray(result_table, dtype=np.float64)
    expected_cols = ["rho", "v", "w", "kappa", "sigma", "h", "misfit"]

    if result_table.ndim != 2 or result_table.shape[1] < 7:
        raise ValueError("`result_table` must be a 2-D array with at least 7 columns")

    if result_table.shape[0] == 0:
        tt_best = DataFrame(columns=expected_cols)
        mt_best = np.empty((0, 6))
        if asnumpy:
            return tt_best.to_numpy(), mt_best
        else:
            return tt_best, mt_best

    min_idx = int(np.argmin(result_table[:, 6]))
    tt_best = DataFrame(result_table[min_idx:min_idx+1, :7], columns=expected_cols)

    mt_best = tt_to_mij(
        tt_best["rho"].to_numpy(),
        tt_best["v"].to_numpy(),
        tt_best["w"].to_numpy(),
        tt_best["kappa"].to_numpy(),
        tt_best["sigma"].to_numpy(),
        tt_best["h"].to_numpy(),
        mtype,
    )

    if asnumpy:
        return tt_best.to_numpy(), mt_best
    else:
        return tt_best, mt_best

def get_accepted_from_result_table(result_table, tolerance, asnumpy=True, mtype="NED"):
    result_table = np.asarray(result_table, dtype=np.float64)
    expected_cols = ["rho", "v", "w", "kappa", "sigma", "h", "misfit"]

    if result_table.ndim != 2 or result_table.shape[1] < 7:
        raise ValueError("`result_table` must be a 2-D array with at least 7 columns")

    accepted = result_table[result_table[:, 6] <= tolerance] if result_table.shape[0] else np.empty((0, 7))

    if accepted.shape[0] == 0:
        tt_accept = DataFrame(columns=expected_cols)
        mt_accept = np.empty((0, 6))
        if asnumpy:
            return tt_accept.to_numpy(), mt_accept
        else:
            return tt_accept, mt_accept

    tt_accept = DataFrame(accepted[:, :7], columns=expected_cols)
    mt_accept = tt_to_mij(
        tt_accept["rho"].to_numpy(),
        tt_accept["v"].to_numpy(),
        tt_accept["w"].to_numpy(),
        tt_accept["kappa"].to_numpy(),
        tt_accept["sigma"].to_numpy(),
        tt_accept["h"].to_numpy(),
        mtype,
    )

    if asnumpy:
        return tt_accept.to_numpy(), mt_accept
    else:
        return tt_accept, mt_accept

def get_best_lunespace_from_result_table(result_table, asnumpy=True, mtype="NED"):
    result_table = np.asarray(result_table, dtype=np.float64)
    expected_cols = ["rho", "v", "w", "kappa", "sigma", "h", "misfit"]

    if result_table.ndim != 2 or result_table.shape[1] < 7:
        raise ValueError("`result_table` must be a 2-D array with at least 7 columns")

    if result_table.shape[0] == 0:
        tt_best_lune = DataFrame(columns=expected_cols)
        mt_best_lune = np.empty((0, 6))
        if asnumpy:
            return tt_best_lune.to_numpy(), mt_best_lune
        else:
            return tt_best_lune, mt_best_lune

    table = DataFrame(result_table[:, :7], columns=expected_cols)
    idx = table.groupby(["v", "w"], sort=False)["misfit"].idxmin()
    tt_best_lune = table.loc[idx].reset_index(drop=True)

    mt_best_lune = tt_to_mij(
        tt_best_lune["rho"].to_numpy(),
        tt_best_lune["v"].to_numpy(),
        tt_best_lune["w"].to_numpy(),
        tt_best_lune["kappa"].to_numpy(),
        tt_best_lune["sigma"].to_numpy(),
        tt_best_lune["h"].to_numpy(),
        mtype,
    )

    if asnumpy:
        return tt_best_lune.to_numpy(), mt_best_lune
    else:
        return tt_best_lune, mt_best_lune

def mt2omega(mt1, mts):

    mt1full = get_full_MT_array(mt1)
    out = np.empty((mts.shape[0], 1), dtype=np.float64)
    for i in range(mts.shape[0]):
        mt2full = get_full_MT_array(mts[i])
        cosom = np.tensordot(mt1full, mt2full)/(np.linalg.norm(mt1full)*np.linalg.norm(mt2full))
        if cosom >1:
            cosom=1
        elif cosom <-1:
            cosom=-1
        out[i] = np.rad2deg(np.arccos(cosom))

    return out