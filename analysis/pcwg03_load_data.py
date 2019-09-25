import copy

import pcwg03_config as pc
import pcwg03_slice_df as psd

# group each meta data category into groups, for plotting histograms
for key, value in pc.meta_var_grouped.items():
    psd.group_meta_element_in_range(key, value)

