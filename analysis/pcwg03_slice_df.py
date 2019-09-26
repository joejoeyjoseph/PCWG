import os
import glob
import copy
import pandas as pd
import numpy as np
import itertools

import pcwg03_config as pc
import pcwg03_convert_df as pcd
import pcwg03_energy_fraction as pef

# code directory
py_file_path = os.getcwd()
# put generated plots near code
out_plot_path = py_file_path+'/plots/'

data_file = glob.glob(pc.data_file_path+'*.xls')

meta_df = pcd.get_metadata_df(data_file)

error_df, extra_error_df = pcd.filter_base_bin_nme(pcd.get_error_df_dict(data_file),
                                                   pcd.get_extra_error_df_dict(data_file))

base_bin_e_df = error_df['base_bin_e']
base_total_e_df = error_df['base_total_e']

def get_base_e_df(error_cat):
    """Load Baseline data frames."""

    nme_bin_e = base_bin_e_df.loc[(((base_bin_e_df['error_cat'] == 'by_range')
                                    & (base_bin_e_df['bin_name'] == 'Outer'))
                                   | (base_bin_e_df['error_cat'] == error_cat))
                                  & (base_bin_e_df['error_name'] == 'nme')]

    dc_total_e = base_total_e_df.loc[((base_total_e_df['error_cat'] == 'overall')
                                              | ((base_total_e_df['error_cat'] == 'by_range')
                                                 & (base_total_e_df['bin_name'] == 'Outer'))
                                              | (base_total_e_df['error_cat'] == error_cat))
                                             & (base_total_e_df['error_name'] == 'data_count')]

    nme_total_e = base_total_e_df.loc[(((base_total_e_df['error_cat'] == 'by_range')
                                        & (base_total_e_df['bin_name'] == 'Outer'))
                                       | (base_total_e_df['error_cat'] == error_cat))
                                      & (base_total_e_df['error_name'] == 'nme')]

    return nme_bin_e, dc_total_e, nme_total_e

def get_base_total_e(error_cat):
    """Load Baseline total error data frame."""

    base_total_e = base_total_e_df.loc[((base_total_e_df['error_cat'] == error_cat)
                                        | (base_total_e_df['error_cat'] == 'by_range'))
                                       & (base_total_e_df['error_name'] == 'nme')]

    return base_total_e

def get_error_in_bin(df, sheet, by_bin, error_name):

    return df[sheet].loc[(df[sheet]['error_cat'] == by_bin) & (df[sheet]['error_name'] == error_name)]

def get_outer_range_nme(df):

    return df.loc[(df['error_cat'] == 'by_range') & (df['bin_name'] == 'Outer') & (df['error_name'] == 'nme')]

def get_wsti_outer_nme(sheet, error_cat):

    out_df = error_df[sheet+'total_e'].loc[((error_df[sheet+'total_e']['error_cat'] == error_cat)
                                            | (error_df[sheet+'total_e']['bin_name'] == 'Outer'))
                                           & (error_df[sheet+'total_e']['error_name'] == 'nme')]

    return out_df

def get_sheet_wsti_range_all_total_e(sheet):
    """Load NME data frame of WS-TI, Inner-Outer Range, and Overall bins."""

    sheet_i = sheet+'total_e'

    error_cat = 'by_ws_ti'

    nme_df = error_df[sheet_i].loc[((error_df[sheet_i]['error_cat'] == error_cat)
                                    | (error_df[sheet_i]['error_cat'] == 'by_range')
                                    | (error_df[sheet_i]['error_cat'] == 'overall'))
                                   & (error_df[sheet_i]['error_name'] == 'nme')]

    ef_df = pef.cal_wsti_ef(error_cat)

    problem_file = pef.check_problematic_file(ef_df, error_cat)

    out_df = pef.remove_problematic_files(nme_df, problem_file, error_cat, pc.wsti_new_bin)

    return out_df

def cal_average_spread(df, u_bin, average_df, spread_df, sheet, rr_choice=pc.robust_resistant_choice):
    """Calculate average and spread statistics for data frame."""

    average = np.empty(len(u_bin))
    spread = np.empty(len(u_bin))

    if rr_choice is None:

        for idx, val in enumerate(u_bin):

            average[idx] = df.loc[df['bin_name'] == val]['error_value'].mean() * 100.
            spread[idx] = df.loc[df['bin_name'] == val]['error_value'].std() * 100.

    else:

        for idx, val in enumerate(u_bin):

            average[idx] = df.loc[df['bin_name'] == val]['error_value'].median() * 100.

            q1 = df.loc[df['bin_name'] == val]['error_value'].quantile(0.25)
            q3 = df.loc[df['bin_name'] == val]['error_value'].quantile(0.75)
            spread[idx] = (q3 - q1) * 100.

    average_df[sheet] = average
    spread_df[sheet] = spread

def strip_df(average_df, spread_df, sheet):

    average_df.rename(columns={sheet: sheet.rstrip('_')},
                      inplace=True)
    spread_df.rename(columns={sheet: sheet.rstrip('_')},
                     inplace=True)

def strip_df_add_file_count(sheet, df, u_bin):

    average_df.rename(columns={sheet: sheet.rstrip('_') + ': ' + str(round(len(df) / len(u_bin)))},
                      inplace=True)
    spread_df.rename(columns={sheet: sheet.rstrip('_') + ': ' + str(round(len(df) / len(u_bin)))},
                     inplace=True)

def find_unique_bin_create_dum(series):

    u_bin = series.unique()

    average = np.empty(len(u_bin))
    spread = np.empty(len(u_bin))

    return (u_bin, average, spread)

def get_wsti_nme_stat():
    """Get average and spread statistics for WS-TI bins."""

    all_wsti_nme_df = pd.DataFrame()

    for idx, sheet in enumerate(pc.matrix_sheet_name_short):

        wsti_nme_df = get_sheet_wsti_range_all_total_e(sheet)

        u_bin = wsti_nme_df['bin_name'].unique()

        if idx == 0:

            average_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])
            spread_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])

        cal_average_spread(wsti_nme_df, u_bin, average_df, spread_df, sheet)

        strip_df(average_df, spread_df, sheet)

        ws_ti_df_toadd = copy.copy(wsti_nme_df)
        ws_ti_df_toadd['method'] = sheet.rstrip('_')
        all_wsti_nme_df = pd.concat([all_wsti_nme_df, ws_ti_df_toadd], axis=0)
        all_wsti_nme_df.reset_index(inplace=True, drop=True)

    return average_df, spread_df, all_wsti_nme_df

def get_outws_nme_stat():
    """Get average and spread statistics for Outer Range WS bins."""

    all_outws_nme_df = pd.DataFrame()

    for method in pc.matrix_sheet_name_short:

        df = error_df[method + 'total_e']
        df_s = df.loc[(df['error_cat'] == 'by_ws_bin_outer') & (df['error_name'] == 'nme')]
        df_s_toadd = copy.copy(df_s)
        df_s_toadd['method'] = method.rstrip('_')
        all_outws_nme_df = pd.concat([all_outws_nme_df, df_s_toadd], axis=0)

    all_outws_nme_df.reset_index(inplace=True, drop=True)

    return all_outws_nme_df

def sort_plot_wsti_df_index(df):
    """Sort WS-TI bin order for plotting."""

    df.rename(index={'ALL': 'Overall'}, inplace=True)
    df = df.reindex(index=pc.sort_wsti_index)

    return df

def group_meta_element_in_range(key, value):
    """Combine meta data into groups for plotting grouped histograms."""

    series_to_edit = meta_df[key]

    if key == 'turbi_spower':  # change units for specific power

        series_to_edit = series_to_edit * 1e3

    tickmark_lim = np.linspace(series_to_edit.min(), series_to_edit.max(), 10)

    if key == 'turbi_rated_power':
        round_d = 3
    elif key == 'turbi_d_hh_ratio':
        round_d = 2
    else:
        round_d = 0

    series_edited = ['' for x in range(len(series_to_edit))]

    for i in range(len(series_to_edit)):
        for j in range(len(tickmark_lim) - 1):
            if series_to_edit[i] >= tickmark_lim[j] and series_to_edit[i] <= tickmark_lim[j + 1]:

                if key == 'turbi_d_hh_ratio':

                    tickmark_start = str(np.round(tickmark_lim[j], round_d))
                    tickmark_end = str(np.round(tickmark_lim[j + 1], round_d))

                else:

                    tickmark_start = str(np.round(tickmark_lim[j], round_d))[:-2]
                    tickmark_end = str(np.round(tickmark_lim[j + 1], round_d))[:-2]

                if (series_to_edit.max() >= 100 and tickmark_lim[j] < 100):

                    tickmark_start = '0' + tickmark_start  # add 0 for sorting

                series_edited[i] = tickmark_start + ' - ' + tickmark_end

            if np.isnan(series_to_edit[i]):

                series_edited[i] = str(np.nan)

    meta_df[value] = series_edited

def remove_0_in_label(df):
    """Remove 0 in the beginning of a string for plotting."""

    x_sorted = df['index'].sort_values()
    x_nozero = copy.copy(x_sorted)
    x_nozero = x_nozero.reset_index()

    if isinstance(x_sorted[0], str) and x_sorted[0][0] == '0':

        for idx, val in enumerate(x_sorted):

            if val[0] == '0':

                with pd.option_context('mode.chained_assignment', None):

                    x_nozero['index'][idx] = x_nozero['index'][idx][1:]  # remove 0 in string

    return x_sorted, x_nozero

def get_outer_meta(error, meta_var, bt_c, y_var):
    """Find error for each method, calculate difference from Baseline.
    Correlate error with meta data variables, if they are numerically represented.
    """

    lump_df = pd.DataFrame()
    lump_corr = np.zeros(0)

    for i, sheet in enumerate(pc.matrix_sheet_name_short):

        outer = error_df[sheet+bt_c].loc[(error_df[sheet+bt_c]['error_cat'] == 'by_range')
                                         & (error_df[sheet+bt_c]['bin_name'] == 'Outer')
                                         & (error_df[sheet+bt_c]['error_name'] == error)]

        base = error_df['base_'+bt_c].loc[(error_df['base_'+bt_c]['error_cat'] == 'by_range')
                                          & (error_df['base_'+bt_c]['bin_name'] == 'Outer')
                                          & (error_df['base_'+bt_c]['error_name'] == error)]

        with pd.option_context('mode.chained_assignment', None):

            if sheet == 'base_':

                outer['diff'] = np.NaN

            # calculate difference between correction methods and Baseline
            else:

                outer['diff'] = (abs(outer['error_value']) - abs(base['error_value'])) * 100

            outer['sheet'] = str(sheet)[:-1]

        outer_all = pd.merge(outer, meta_df, on='file_name')

        if all(isinstance(x, (float, int)) for x in meta_df[meta_var]): # if meta x-axis is numeric

            corr = np.corrcoef(list(outer_all[y_var].values), list(outer_all[meta_var].values))

            if not math.isnan(corr[0][1]):

                lump_corr = np.append(lump_corr, round(corr[0][1], 2))

        lump_df = pd.concat([lump_df, outer_all], sort=True)

    return lump_df, lump_corr

def get_nme_diff_range():

    outer_base_te_df = get_outer_range_nme(base_total_e_df)
    outer_dt_te_df = get_outer_range_nme(error_df['den_turb_total_e'])
    outer_d2_te_df = get_outer_range_nme(error_df['den_2dpdm_total_e'])
    outer_dat_te_df = get_outer_range_nme(error_df['den_augturb_total_e'])
    outer_d3_te_df = get_outer_range_nme(error_df['den_3dpdm_total_e'])

    for idx, file in enumerate(outer_base_te_df['file_name'].unique()):

        base_nme = outer_base_te_df.loc[outer_base_te_df['file_name'] == file]['error_value'].values[0] * 100.
        dt_nme = outer_dt_te_df.loc[outer_dt_te_df['file_name'] == file]['error_value'].values[0] * 100.
        d2_nme = outer_d2_te_df.loc[outer_d2_te_df['file_name'] == file]['error_value'].values[0] * 100.
        dat_nme = outer_dat_te_df.loc[outer_dat_te_df['file_name'] == file]['error_value'].values[0] * 100.
        d3_nme = outer_d3_te_df.loc[outer_d3_te_df['file_name'] == file]['error_value'].values[0] * 100.

        method_nme_list = np.array([abs(dt_nme), abs(d2_nme), abs(dat_nme), abs(d3_nme)])

        nme_diff = method_nme_list - abs(base_nme)

        if idx == 0:
            nme_diff_list = nme_diff

        else:
            nme_diff_list = np.vstack((nme_diff_list, nme_diff))

    nme_diff_df = pd.DataFrame(nme_diff_list.T)

    nme_range = nme_diff_df.max() - nme_diff_df.min()

    improve_outer_list = ['Mixed'] * nme_diff_df.shape[1]

    for col in range(nme_diff_df.shape[1]):

        if all(item > 0 for item in nme_diff_df[col]):
            improve_outer_list[col] = 'Worse'

        elif all(item < 0 for item in nme_diff_df[col]):
            improve_outer_list[col] = 'Improved'

    nme_range_p_df = pd.DataFrame({'nme': nme_range, 'all': improve_outer_list})
    nme_range_p_df.reset_index(inplace=True)

    return nme_diff_df, nme_range_p_df
