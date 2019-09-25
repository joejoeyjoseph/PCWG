import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import numpy as np
import pandas as pd
import math

import geopandas as gpd
from descartes import PolygonPatch

import pcwg03_config as pc
import pcwg03_convert_df as pcd
import pcwg03_energy_fraction as pef
import pcwg03_slice_df as psd

meta_df = psd.meta_df

save_fig = pc.save_fig

dpi_choice = 600 # output plot resolution

py_file_path = os.getcwd()
# put generated plots near code
out_plot_path = py_file_path+'/plots/'

fs = 12
f15 = 15

plt.rcParams.update({'font.size': fs})

xp_f1, yp_f1 = 0.04, 0.93 # x, y positions

def save_plot(sub_dir, var, plot_type, pdf=True):
    """Export figure to either pdf or png file."""

    if not os.path.exists(out_plot_path+'/'+ sub_dir):
        os.makedirs(out_plot_path+'/'+sub_dir)

    if pdf is True:

        plt.savefig(out_plot_path+sub_dir+'/'+var+'_'+plot_type+'.pdf',
                    bbox_inches='tight', dpi=dpi_choice)

    else:

        plt.savefig(out_plot_path+'/'+sub_dir+'/'+var+'_'+plot_type+'.png',
                    bbox_inches='tight', dpi=dpi_choice)

def finish_plot(sub_dir, var, plot_type, tight_layout=True, save_fig=False, pdf=True):
    """Terminating procedures for plotting."""

    if tight_layout is True:

        plt.tight_layout()

    if save_fig is True:

        save_plot(sub_dir, var, plot_type, pdf)

    plt.show()

def plot_wsti_energy_fraction_box():
    """Plot 4 box plots for WS-TI, ITI-OS, and Inner-Outer Ranges.
    A pair of box plots on energy and data fractions, and a pair box plots on NMEs.
    Similar to `plot_outws_energy_fraction_box`.
    """

    ef_filter_df1, ef_filter_df2, wsti_nme_df1, wsti_nme_df2 = pef.get_wsti_ef_nme()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})

    ax1 = sns.boxplot(x='bin_name', y='value', data=ef_filter_df1, hue='error_name',
                      palette='colorblind', ax=ax1)
    ax1 = sns.swarmplot(x='bin_name', y='value', data=ef_filter_df1, hue='error_name',
                        alpha=0.7, dodge=True, ax=ax1)
    ax1.set_ylabel('Fraction (%)')
    ax1.set_xlabel('')
    ax1.set_ylim([-5, 95])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], ['Data fraction', 'Energy fraction'], loc='upper left')

    ax2 = sns.boxplot(x='bin_name', y='value', data=ef_filter_df2, hue='error_name',
                      palette='colorblind', ax=ax2)
    ax2 = sns.swarmplot(x='bin_name', y='value', data=ef_filter_df2, hue='error_name',
                        alpha=0.7, dodge=True, ax=ax2)
    ax2.legend_.remove()
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_ylim([-5, 95])
    ax2.set_yticklabels([])

    ax3 = sns.boxplot(x='bin_name', y='error_value', data=wsti_nme_df1, hue='error_name',
                      palette='BuGn_r', ax=ax3)
    ax3.axhline(0, ls='--', color='grey')
    ax3.set_ylabel('NME (%)')
    ax3.set_xlabel('WS-TI bins')
    ax3.legend_.remove()
    ax3.set_ylim([-2.5, 2.5])

    ax4 = sns.boxplot(x='bin_name', y='error_value', data=wsti_nme_df2, hue='error_name',
                      palette='BuGn_r', ax=ax4)
    ax4.axhline(0, ls='--', color='grey')
    ax4.set_ylabel('')
    ax4.set_xlabel('Inner-Outer Range')
    ax4.legend_.remove()
    ax4.set_ylim([-2.5, 2.5])
    ax4.set_yticklabels([])

    ax1.text(0.94, 0.9, '(a)', color='k', fontsize=12, transform=ax1.transAxes)
    ax2.text(0.82, 0.9, '(b)', color='k', fontsize=12, transform=ax2.transAxes)
    ax3.text(0.94, 0.9, '(c)', color='k', fontsize=12, transform=ax3.transAxes)
    ax4.text(0.82, 0.9, '(d)', color='k', fontsize=12, transform=ax4.transAxes)

    finish_plot('meta', 'wsti_energyfraction', 'box')

def plot_outws_energy_fraction_box():
    """Plot 2 box plots for Outer Range WS.
    A box plot on energy and data fractions, and a box plot on NMEs.
    Similar to `plot_wsti_energy_fraction_box`.
    """

    dc_ef_all_df_s, outws_nme_df_s = pef.get_outws_ef_nme()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1 = sns.boxplot(x='bin_name', y='value', data=dc_ef_all_df_s, hue='error_name',
                      palette='colorblind', ax=ax1)

    ax1.set_ylabel('Fraction (%)', fontsize=f15)
    ax1.set_xlabel('')

    ax1.tick_params(labelsize=f15)
    ax1.xaxis.set_tick_params(rotation=45)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], ['Data fraction', 'Energy fraction'], fontsize=f15)

    ax2 = sns.boxplot(x='bin_name', y='error_value', data=outws_nme_df_s, hue='error_name',
                      palette='BuGn_r', ax=ax2)

    ax2.axhline(0, ls='--', color='grey')
    ax2.set_ylabel('NME (%)', fontsize=f15)
    ax2.set_xlabel('Normalized wind speed category', fontsize=f15)
    ax2.set_ylim([-1.2, 1.2])
    ax2.legend_.remove()

    ax2.tick_params(labelsize=f15)
    ax2.xaxis.set_tick_params(rotation=45)

    ax1.text(0.94, 0.91, '(a)', color='k', fontsize=f15, transform=ax1.transAxes)
    ax2.text(0.94, 0.91, '(b)', color='k', fontsize=f15, transform=ax2.transAxes)

    finish_plot('meta', 'outws_energyfraction', 'box')

def plot_hist_series(df, var, name):
    """Plot histogram for series of meta data."""

    p_series = pd.Series(df[var])

    p_series.value_counts(dropna=False).plot(kind='bar', rot=45)
    plt.ylabel('Count')
    plt.title(name)

    finish_plot('meta', 'meta_' + var, 'hist')

def loop_meta_hist():
    """Generate histograms from available meta data."""

    for var, name in zip(pc.meta_var_names_turb, pc.meta_xls_names_turb):
        plot_hist_series(psd.meta_df, var, name)

def plot_group_meta_hist():
    """Plot 4 histograms using grouped bins on x-axis."""

    hist_df1 = pd.DataFrame({'turbi_dia_grouped': meta_df['turbi_dia_grouped'].value_counts(dropna=False)})
    hist_df1.reset_index(inplace=True)
    hist_df1.replace('143 - 154', '120+', inplace=True)
    x_sorted1, x_nozero1 = psd.remove_0_in_label(hist_df1)

    hist_df2 = pd.DataFrame({'turbi_hh_grouped': meta_df['turbi_hh_grouped'].value_counts(dropna=False)})
    hist_df2.reset_index(inplace=True)
    hist_df2.replace('132 - 143', '110+', inplace=True)
    hist_df2.replace('110 - 121', '110+', inplace=True)
    hist_df22 = hist_df2.groupby(['index'], as_index=False).agg('sum')
    x_sorted2, x_nozero2 = psd.remove_0_in_label(hist_df22)

    hist_df3 = pd.DataFrame({'turbi_spower_grouped': meta_df['turbi_spower_grouped'].value_counts(dropna=False)})
    hist_df3.reset_index(inplace=True)
    hist_df3.replace('489 - 536', '441+', inplace=True)
    hist_df3.replace('536 - 583', '441+', inplace=True)
    hist_df33 = hist_df3.groupby(['index'], as_index=False).agg('sum')
    x_sorted3, x_nozero3 = psd.remove_0_in_label(hist_df33)

    meta_var_names_array = np.array(pc.meta_var_names)
    year_measure_idx = np.where(meta_var_names_array == 'year_measuremt')
    year_str = pc.meta_var_names[int(year_measure_idx[0])]

    hist_df4 = pd.DataFrame({year_str: meta_df[year_str].value_counts(dropna=False)})
    hist_df4.reset_index(inplace=True)
    x_sorted4, x_nozero4 = psd.remove_0_in_label(hist_df4)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    sns.barplot(x='index', y='turbi_dia_grouped', data=hist_df1, ax=ax1, order=x_sorted1, color='c')
    ax1.set_xticklabels(labels=x_nozero1['index'], rotation=45)
    ax1.set_xlabel('Turbine rotor diameter (m)', fontsize=fs + 1)
    ax1.set_ylabel('Count', fontsize=fs + 1)

    sns.barplot(x='index', y='turbi_hh_grouped', data=hist_df22, ax=ax2, order=x_sorted2, color='c')
    ax2.set_xticklabels(labels=x_nozero2['index'], rotation=45)
    ax2.set_xlabel('Turbine hub height (m)', labelpad=10, fontsize=fs + 1)
    ax2.set_ylabel('')

    sns.barplot(x='index', y='turbi_spower_grouped', data=hist_df33, ax=ax3, order=x_sorted3, color='c')
    ax3.set_xticklabels(labels=x_nozero3['index'], rotation=45)
    ax3.set_xlabel(r'Turbine specific power (W m$^{-2}$)', fontsize=fs + 1)
    ax3.set_ylabel('Count', fontsize=fs + 1)

    sns.barplot(x='index', y=year_str, data=hist_df4, ax=ax4, order=x_sorted4, color='c')
    ax4.set_xticklabels(labels=x_nozero4['index'], rotation=45)
    ax4.set_xlabel('Year of measurement', labelpad=25, fontsize=fs + 1)
    ax4.set_ylabel('')

    ax1.text(xp_f1, yp_f1, '(a)', color='k', fontsize=fs, transform=ax1.transAxes)
    ax2.text(xp_f1, yp_f1, '(b)', color='k', fontsize=fs, transform=ax2.transAxes)
    ax3.text(xp_f1, yp_f1, '(c)', color='k', fontsize=fs, transform=ax3.transAxes)
    ax4.text(xp_f1, yp_f1, '(d)', color='k', fontsize=fs, transform=ax4.transAxes)

    finish_plot('meta', 'meta_', 'hist_ranked')

def plot_map():
    """Map submission origins, if available."""

    country_series = meta_df['geog_country'].value_counts(dropna=False)
    country_series = country_series.rename_axis('country').reset_index()
    country_na = country_series.loc[country_series['country'].isnull()].index
    country_series_plot = country_series.drop(country_na[0])  # drop NaN

    # count of NaN
    nan_country = str(country_series.loc[country_series['country'].isnull()]['geog_country'][0])
    total_country = str(len(meta_df['geog_country']))

    colmap = plt.cm.get_cmap('viridis')
    colmap(1)

    country_num_max = max(country_series_plot['geog_country'])

    country_num = np.linspace(0, 1, country_num_max + 1)

    country_num_on_map = country_num[country_series_plot['geog_country']]

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def plotCountryPatch(axes, country_name, fcolor):
        # plot a country on the provided axes
        nami = world[world.name == country_name]
        namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
        namig0 = {'type': namigm[0]['geometry']['type'], \
                  'coordinates': namigm[0]['geometry']['coordinates']}
        axes.add_patch(PolygonPatch(namig0, fc=fcolor, ec='black', alpha=0.85, zorder=2))

    colmap = plt.cm.get_cmap('viridis')

    ax = world.plot(figsize=(8, 4), edgecolor=u'gray', color='w')

    for x, y in zip(country_series_plot['country'].values, country_num_on_map):
        plotCountryPatch(ax, x, colmap(y))

    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title(nan_country + ' of ' + total_country + ' submissions have unknown countries')

    fig = ax.get_figure()
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap=colmap)
    sm._A = []
    cbr = fig.colorbar(sm, cax=cax)
    cbr.set_ticks(np.linspace(0, 1, country_num_max))
    cbr.ax.set_yticklabels(np.linspace(1, country_num_max, country_num_max))

    finish_plot('meta', 'meta', 'map', tight_layout=False)

def plot_NME_hist():
    """Plot NME histograms: pre-NME-filtering and post-NME-filtering."""

    plt.figure(1, figsize=(8, 6))

    gridspec.GridSpec(2, 2)

    # need to get pre-nme-filter error data frame
    df1 = pcd.get_error_df_dict(psd.data_file)['base_total_e']

    df1p = df1.loc[(df1['error_cat'] == 'by_range') & (df1['error_name'] == 'nme')]

    inner_nme = df1p['error_value'].loc[df1p['bin_name'] == 'Inner'] * 100
    outer_nme = df1p['error_value'].loc[df1p['bin_name'] == 'Outer'] * 100

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

    a_value = 0.7

    sns.distplot(list(outer_nme.values), color='#73c0c4', label='Outer Range', bins=10, kde=False, ax=ax1,
                 hist_kws={'alpha': a_value})
    sns.distplot(list(inner_nme.values), color='#3c758b', label='Inner Range', bins=5, kde=False, ax=ax1,
                 hist_kws={'alpha': a_value})

    plt.ylabel('Count')
    plt.xlabel('NME (%)')

    plt.legend()
    plt.tight_layout()

    sheet_bt_choice = 'base_total_e'
    df2 = psd.error_df

    def choose_in_out_def(in_or_out, in_out_def):

        selection = ((df2[sheet_bt_choice]['error_cat'] == 'by_range')
                     & (df2[sheet_bt_choice]['error_name'] == 'nme')
                     & (df2[sheet_bt_choice]['bin_name'] == in_or_out)
                     & (df2[sheet_bt_choice]['file_name']
                        .isin(meta_df.loc[meta_df['inner_def'] == in_out_def]['file_name'])))

        return selection

    inner_a = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'A')]['error_value'] * 100
    inner_b = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'B')]['error_value'] * 100
    inner_c = df2[sheet_bt_choice].loc[choose_in_out_def('Inner', 'C')]['error_value'] * 100

    outer_a = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'A')]['error_value'] * 100
    outer_b = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'B')]['error_value'] * 100
    outer_c = df2[sheet_bt_choice].loc[choose_in_out_def('Outer', 'C')]['error_value'] * 100

    p23_c = ['seagreen', 'limegreen', 'lawngreen']
    # p23_c = ['red', 'darkorange', 'gold']

    ax2 = plt.subplot2grid((2, 2), (1, 0))

    ax2.hist([inner_a, inner_b, inner_c], stacked=True, color=p23_c)
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Filtered Inner Range NME (%)')

    ax3 = plt.subplot2grid((2, 2), (1, 1))

    ax3.hist([outer_a, outer_b, outer_c], stacked=True, color=p23_c)
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Filtered Outer Range NME (%)')

    labels = ['A', 'B', 'C']
    plt.legend(labels, title='Definition', loc='upper left')

    ax1.text(0.03, 0.89, '(a)', color='k', fontsize=12, transform=ax1.transAxes)
    ax2.text(0.05, 0.88, '(b)', color='k', fontsize=12, transform=ax2.transAxes)
    ax3.text(0.89, 0.88, '(c)', color='k', fontsize=12, transform=ax3.transAxes)

    finish_plot('error_hist', 'nme', '3def_hist')

def plot_wsti_nme_box():
    """Plot 4 panel box plots for WS-TI NME."""

    box_plot_y_scale = 0.5  # zoom in
    # box_plot_y_scale = 1

    def ws_ti_df_by_sheet(sheet_name, sheet_name_short, df, bt_choice, error_name, file_num):

        sheet_bt_choice = sheet_name_short + bt_choice

        ws_ti_df = df[sheet_bt_choice].loc[(df[sheet_bt_choice]['error_cat'] == 'by_ws_ti')
                                           & (df[sheet_bt_choice]['error_name'] == error_name)]

        if file_num is True:
            sheet_name_end = ': ' + str(round(len(ws_ti_df) / 4))
        else:
            sheet_name_end = ''

        ws_ti_df.insert(0, 'sheet_name', str(sheet_name_short)[:-1] + sheet_name_end)

        return ws_ti_df

    def loop_box_plot(bt_choice, error_name, error_df, file_num_choice=False, extra_error_df=None):

        for i, i_short in zip(pc.matrix_sheet_name, pc.matrix_sheet_name_short):

            dum_df = ws_ti_df_by_sheet(i, i_short, error_df, bt_choice, error_name, file_num=file_num_choice)

            if pc.matrix_sheet_name.index(i) == 0:

                ws_ti_df = dum_df

            else:

                ws_ti_df = ws_ti_df.append(dum_df)

        if extra_error_df is not None:

            for i, i_short in zip(pc.correction_list, pc.extra_matrix_sheet_name_short):

                dum_df = ws_ti_df_by_sheet(i, i_short, extra_error_df, bt_choice, error_name,
                                           file_num=file_num_choice)

                if pc.correction_list.index(i) == 0:

                    ws_ti_df_extra = dum_df

                else:

                    ws_ti_df_extra = ws_ti_df_extra.append(dum_df)

            ws_ti_df = ws_ti_df.append(ws_ti_df_extra)

        ws_ti_df['error_value'] = ws_ti_df['error_value'].astype(float) * 100

        ws_ti_error_min, ws_ti_error_max = ws_ti_df['error_value'].min(), ws_ti_df['error_value'].max()
        ws_ti_error_abs_max = np.max([abs(ws_ti_error_min), abs(ws_ti_error_max)])

        lws_lti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'LWS-LTI']
        lws_hti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'LWS-HTI']
        hws_lti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'HWS-LTI']
        hws_hti_df = ws_ti_df.loc[ws_ti_df['bin_name'] == 'HWS-HTI']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

        def plot_box_by_sheet(df, ax, sub_t):

            # add grey to colorblind... manually
            # sns.color_palette(['grey'])
            # sns.color_palette('colorblind')

            new_p = [(0.5019607843137255, 0.5019607843137255, 0.5019607843137255),
                     (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                     (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                     (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                     (0.8352941176470589, 0.3686274509803922, 0.0),
                     (0.8, 0.47058823529411764, 0.7372549019607844),
                     (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                     (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                     (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                     (0.9254901960784314, 0.8823529411764706, 0.2),
                     (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)]

            ax = sns.boxplot(x='sheet_name', y='error_value', data=df, ax=ax, palette=new_p)
            # ax = sns.boxplot(x='sheet_name', y='error_value', data=df, ax=ax, palette='colorblind')
            # ax = sns.swarmplot(x='sheet_name', y='error_value', data=df, ax=ax, palette='colorblind')

            if error_name == 'nme':

                ax.axhline(0, ls='--', color='grey')

            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.set_title(df['bin_name'].iloc[0])
            ax.set_ylabel(bt_choice + 'nergy ' + error_name + ' (%)')

            # ax.set_ylim([ws_ti_error_min*box_plot_y_scale, ws_ti_error_max*box_plot_y_scale])
            ax.set_ylim([-ws_ti_error_abs_max * box_plot_y_scale, ws_ti_error_abs_max * box_plot_y_scale])

            ax.text(0.95, 0.92, sub_t, color='k', fontsize=12, transform=ax.transAxes)

            return ax

        plot_box_by_sheet(lws_lti_df, ax1, '(a)')
        plot_box_by_sheet(lws_hti_df, ax2, '(b)')
        plot_box_by_sheet(hws_lti_df, ax3, '(c)')
        plot_box_by_sheet(hws_hti_df, ax4, '(d)')

        if extra_error_df is not None:
            var = 'wsti_nme_boxplot_extra'
        else:
            var = 'wsti_nme_boxplot'

        finish_plot('results', var, bt_choice + '_' + error_name)

    loop_box_plot('total_e', 'nme', psd.error_df, extra_error_df=psd.extra_error_df)

def plot_nme_avg_spread_heatmap(ee_df=None, rr_choice=None):
    """Mass generate heatmaps of NME average and NME spread."""

    def loop_nme_avg_spread_heatmap(bt_choice, error_name, by_bin, e_df=psd.error_df, ee_df=None,
                                    rr_choice=pc.robust_resistant_choice):

        for idx, i_short in enumerate(pc.matrix_sheet_name_short):

            df = psd.get_error_in_bin(e_df, i_short+bt_choice, by_bin, error_name)

            u_bin, average, spread = psd.find_unique_bin_create_dum(df['bin_name'])

            if idx == 0:
                average_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])
                spread_df = pd.DataFrame(index=u_bin, columns=[pc.matrix_sheet_name_short])

            psd.cal_average_spread(df, u_bin, average_df, spread_df, i_short, rr_choice)

        if ee_df is not None:

            for idx, i_short in enumerate(pc.extra_matrix_sheet_name_short):
                df = psd.get_error_in_bin(psd.extra_error_df, i_short+bt_choice, by_bin, error_name)

                u_bin, average, spread = psd.find_unique_bin_create_dum(df['bin_name'])

                psd.cal_average_spread(df, u_bin, average_df, spread_df, i_short, rr_choice)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if error_name == 'nme':

            sns.heatmap(average_df, linewidths=.5, cmap='RdBu_r', center=0, ax=ax1)

        elif error_name == 'nmae':

            sns.heatmap(average_df, linewidths=.5, ax=ax1)

        if rr_choice is None:

            avg_title, spd_title, out_name = 'mean', 'standard deviation', 'mean_sd'

        else:

            avg_title, spd_title, out_name = 'median', 'interquartile range', 'median_iqr'

        ax1.set_xlabel('correction methods')
        ax1.set_ylabel(by_bin)
        ax1.yaxis.set_tick_params(rotation=0)
        ax1.set_title(bt_choice + 'nergy ' + avg_title + ' ' + error_name + ' (%)')

        sns.heatmap(spread_df, linewidths=.5, cmap='viridis', ax=ax2)
        ax2.set_xlabel('correction methods')
        ax2.set_ylabel(by_bin)
        ax2.yaxis.set_tick_params(rotation=0)
        ax2.set_title(bt_choice + 'nergy ' + spd_title + ' of ' + error_name + ' (%)')

        if ee_df is not None:
            var = 'nme_avg_spread_boxplot_extra'
        else:
            var = 'nme_avg_spread_boxplot'

        finish_plot('results', var, bt_choice + '_heatmap')

    for by_bin in range(1, len(pc.error_cat_short)):
        for bt_j in pc.bt_choice:
            for k in range(1, len(pc.error_name)):

                loop_nme_avg_spread_heatmap(bt_j, pc.error_name[k], pc.error_cat_short[by_bin],
                                            e_df=psd.error_df, ee_df=ee_df,
                                            rr_choice=rr_choice)

def plot_inner_outer_data_count_box_hist():
    """Plot Inner Range and Outer Range data count"""

    box_io_df = psd.error_df['base_total_e'].loc[(psd.error_df['base_total_e']['error_cat'] == 'by_range')
                                                 & (psd.error_df['base_total_e']['error_name'] == 'data_count')]

    u_bin = box_io_df['bin_name'].unique()

    for idx, val in enumerate(u_bin):

        data_count = box_io_df.loc[box_io_df['bin_name'] == val]['error_value'].values

        if idx == 0:

            box_dict = {val: data_count}

        else:

            box_dict[val] = data_count

    box_df = pd.DataFrame(box_dict)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(data=box_df, ax=ax1, palette='GnBu_d')
    ax1.set_xticklabels(labels=['Inner Range', 'Outer Range'])

    ax1.set_ylabel('10-minute data count')

    base_inner_count = psd.error_df['base_bin_e'].loc[(psd.error_df['base_bin_e']['error_cat'] == 'by_range')
                                                      & (psd.error_df['base_bin_e']['bin_name'] == 'Inner')
                                                      & (psd.error_df['base_bin_e']['error_name'] == 'data_count')]

    base_outer_count = psd.error_df['base_bin_e'].loc[(psd.error_df['base_bin_e']['error_cat'] == 'by_range')
                                                      & (psd.error_df['base_bin_e']['bin_name'] == 'Outer')
                                                      & (psd.error_df['base_bin_e']['error_name'] == 'data_count')]

    ratio_inout_count = pd.Series([base_outer_count['error_value'].values / base_inner_count['error_value'].values])

    ax2.hist(ratio_inout_count.values, color='grey')
    ax2.set_ylabel('File count')
    ax2.set_xlabel(r'$\mathrm{\mathsf{\frac{Outer\/\/Range\/\/data\/\/count}{Inner\/\/Range\/\/data\/\/count}}}$',
                   fontsize=fs + 5)

    ax1.text(xp_f1, yp_f1, '(a)', color='k', fontsize=fs, transform=ax1.transAxes)
    ax2.text(xp_f1, yp_f1, '(b)', color='k', fontsize=fs, transform=ax2.transAxes)

    finish_plot('meta', 'data_count', 'box_hist')

def plot_file_data_count_hist_box():
    """Mass generate histogram of file count and box plot of data count for each category.
    Check if every sheet/method has the same file count for each inflow category.
    """

    def loop_count_histbox(by_bin, e_df, sheet_name, no_hist=None):

        for idx, i_short in enumerate(sheet_name):

            do_plot = False

            bt_choice = 'bin_e'  # same for total_e

            df = e_df[i_short + bt_choice].loc[(e_df[i_short + bt_choice]['error_cat'] == by_bin)
                                               & (e_df[i_short + bt_choice]['error_name'] == 'data_count')]

            u_bin = df['bin_name'].unique()

            nan_bin_count = np.zeros(len(u_bin))

            for i in range(len(df)):
                for idx, val in enumerate(u_bin):
                    if df.iloc[i]['bin_name'] == val:
                        if pd.isnull(df.iloc[i]['error_value']):
                            nan_bin_count[idx] += 1

            file_num = len(df) / len(u_bin)
            bin_count = file_num - nan_bin_count

            try:
                lump_bin_count  # see if data duplicates already
            except NameError:

                lump_bin_count = bin_count
                lump_bin_count = np.expand_dims(lump_bin_count, axis=0)
                do_plot = True

            if lump_bin_count.shape[0] > 1:
                for i in range(lump_bin_count.shape[0]):
                    if np.array_equal(bin_count, lump_bin_count[i]):
                        do_plot = False
                    else:
                        do_plot = True

            # if data count not duplicate
            if do_plot:

                print(i_short + ' contains unique file count distribution for ' + by_bin)

                file_data = {'d_bin': u_bin, 'd_count': bin_count}

                hist_df = pd.DataFrame(file_data)

                for idx, val in enumerate(u_bin):
                    data_count = df.loc[df['bin_name'] == val]['error_value'].values
                    if idx == 0:
                        box_dict = {val: data_count}
                    else:
                        box_dict[val] = data_count

                box_df = pd.DataFrame(box_dict)

                if no_hist is None:

                    no_hist = ''

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    sns.barplot(x='d_bin', y='d_count', data=hist_df, ax=ax1, color='orchid')
                    ax1.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax1.set_title(str(i_short)[:-1] + ', max: ' + str(int(np.max(bin_count))) + ' files')
                    ax1.set_xlabel(by_bin)
                    ax1.set_ylabel('file count')

                    sns.boxplot(data=box_df, ax=ax2, palette='colorblind')
                    ax2.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax2.set_title(str(i_short)[:-1])
                    ax2.set_xlabel(by_bin)
                    ax2.set_ylabel('data count')

                    if len(hist_df['d_bin']) > 15:  # every other xtick label
                        for label_ax1, label_ax2 in zip(ax1.xaxis.get_ticklabels()[::2],
                                                        ax2.xaxis.get_ticklabels()[::2]):
                            label_ax1.set_visible(False)
                            label_ax2.set_visible(False)

                elif no_hist == 'box':

                    no_hist = no_hist + '_'

                    fig, ax2 = plt.subplots(figsize=(6, 5))

                    sns.boxplot(data=box_df, ax=ax2, palette='colorblind')
                    ax2.set_xticklabels(labels=hist_df['d_bin'], rotation=45)
                    ax2.set_title(str(i_short)[:-1])
                    ax2.set_xlabel(by_bin)
                    ax2.set_ylabel('data count')

                    if len(hist_df['d_bin']) > 15:  # every other xtick label
                        for label_ax2 in ax2.xaxis.get_ticklabels()[::2]:
                            label_ax2.set_visible(False)

                finish_plot('meta', no_hist + by_bin + '_' + str(i_short)[:-1], 'count_hist_box')

                # if i > 1:
                lump_bin_count = np.concatenate((lump_bin_count, np.expand_dims(bin_count, axis=0)), axis=0)

            else:

                print(i_short + ' contains duplicating file count distribution for ' + by_bin)

        del lump_bin_count

    for i in range(1, len(pc.error_cat_short)):

        loop_count_histbox(pc.error_cat_short[i], psd.error_df, pc.matrix_sheet_name_short)
        loop_count_histbox(pc.error_cat_short[i], psd.extra_error_df, pc.extra_matrix_sheet_name_short)

def get_outer_meta(error, meta_var, bt_c, y_var):

    lump_df = pd.DataFrame()
    lump_corr = np.zeros(0)

    for i, sheet in enumerate(pc.matrix_sheet_name_short):

        outer = psd.error_df[sheet+bt_c].loc[(psd.error_df[sheet+bt_c]['error_cat'] == 'by_range')
                                             & (psd.error_df[sheet+bt_c]['bin_name'] == 'Outer')
                                             & (psd.error_df[sheet+bt_c]['error_name'] == error)]

        base = psd.error_df['base_'+bt_c].loc[(psd.error_df['base_'+bt_c]['error_cat'] == 'by_range')
                                              & (psd.error_df['base_'+bt_c]['bin_name'] == 'Outer')
                                              & (psd.error_df['base_'+bt_c]['error_name'] == error)]

        with pd.option_context('mode.chained_assignment', None):

            if sheet == 'base_':
                outer['diff'] = np.NaN
            else:
                outer['diff'] = (abs(outer['error_value']) - abs(base['error_value'])) * 100

            outer['sheet'] = str(sheet)[:-1]

        outer_all = pd.merge(outer, meta_df, on='file_name')

        if all(isinstance(x, (float, int)) for x in meta_df[meta_var]):  # if meta x-axis is numeric

            corr = np.corrcoef(list(outer_all[y_var].values), list(outer_all[meta_var].values))

            if not math.isnan(corr[0][1]):
                lump_corr = np.append(lump_corr, round(corr[0][1], 2))

        lump_df = pd.concat([lump_df, outer_all], sort=True)

    return lump_df, lump_corr

def loop_outer_diff_scatter(meta_var, one_plot=None, diff_choice=None):

    if diff_choice is None:

        y_var = 'error_value'
        y_title_end = ''

    else:

        y_var = 'diff'
        y_title_end = ' - Baseline'

    def add_corr_text(ax, corr):

        add_x = 0.13

        if corr.size != 0:

            ax.text(0.7, 0.94, 'correlation', color='k',
                    weight='semibold', transform=ax.transAxes)  # ratio of axes

            for idx, val in enumerate(corr):
                ax.text(0.75, 1 - add_x, val, color=sheet_color[idx],
                        weight='semibold', transform=ax.transAxes)

                add_x += 0.07

    for bt_c in pc.bt_choice:

        ########################
        ##### SKIPPING ONE #####
        ########################
        if meta_var != 'year_operatn':  # HARD CODED -- no useful data there

            nme_df, nme_corr = get_outer_meta('nme', meta_var, bt_c, y_var)
            nmae_df, nmae_corr = get_outer_meta('nmae', meta_var, bt_c, y_var)

            if diff_choice is not None:

                nme_df = nme_df.loc[~(nme_df['sheet'] == 'base')]
                nmae_df = nmae_df.loc[~(nmae_df['sheet'] == 'base')]

            nme_file_num = str(round(nme_df[meta_var].count()/len(nme_df['sheet'].unique())))[:-2]
            nmae_file_num = str(round(nmae_df[meta_var].count()/len(nmae_df['sheet'].unique())))[:-2]

            if meta_var == 'turbi_spower':  # change units for specific power
                nme_df[meta_var] = nme_df[meta_var] * 1e3
                nmae_df[meta_var] = nmae_df[meta_var] * 1e3

            if one_plot is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            else:
                fig, ax1 = plt.subplots(figsize=(6, 4))

            axx = sns.color_palette(palette='colorblind')
            sheet_color = axx.as_hex()

            if one_plot is None:

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nme_df, alpha=0.5,
                                palette='colorblind', ax=ax1, legend=False)

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nmae_df, alpha=0.5,
                                palette='colorblind', ax=ax2)
                ax2.set_xlabel(meta_var+': '+nmae_file_num+' files')
                ax2.set_ylabel(bt_c+' nmae'+y_title_end)
                ax2.axhline(y=0, linestyle='--', color='grey')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                add_corr_text(ax2, nmae_corr)

                op_text = ''

            else: # plot NMEs only

                sns.scatterplot(x=meta_var, y=y_var, hue='sheet', data=nme_df, alpha=0.5,
                                palette='colorblind', ax=ax1)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                op_text = '_one'

            ax1.set_xlabel(meta_var+': '+nme_file_num+' files')
            ax1.set_ylabel(bt_c+' nme'+y_title_end)
            ax1.axhline(y=0, linestyle='--', color='grey')
            add_corr_text(ax1, nme_corr)

            finish_plot('results', 'scatter_diff_outer', meta_var+op_text+'_'+bt_c)

def plot_outer_diff_scatter(one_plot=None, diff_choice=None):

    for meta_var in pc.meta_var_names_turb:

        loop_outer_diff_scatter(meta_var, one_plot=one_plot, diff_choice=diff_choice)