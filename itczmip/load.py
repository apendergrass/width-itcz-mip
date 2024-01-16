import os
import xarray as xr

def load_data(model, experiment_list):
    root_dir = os.path.join('data', model)
    
    filenames = define_filenames(model)
    dim_dict = define_dimension_rename_dict()
    
    data = {}
    
    for exp in experiment_list:
        filename = os.path.join(root_dir, exp, filenames[exp])
        
        try:
            data_exp = xr.open_dataset(filename)
            data_exp = data_exp.squeeze()
            data_exp.attrs["case"] = exp
            data_exp.attrs["model"] = model
            
            # rename coordinates (lat and plev) so consistent across models
            #data_exp.rename(dim_dict[model], inplace=True)
            data_exp = data_exp.rename(dim_dict[model])
                
            data[exp] = data_exp
        except FileNotFoundError:
            print('Warning: %s for %s was not found.' % (filenames[exp], model))
        
    return data


def define_filenames(model):
    if model == 'CESM1':
        filenames = {'itcz-SST': 'itcz-SST_zonal_mean_climo_years_11_to_20.nc',
                    'itcz-slab': 'itcz-slab_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-m40': 'itcz-slab-m40_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-m20': 'itcz-slab-m20_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-p20': 'itcz-slab-p20_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-p40': 'itcz-slab-p40_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-4xCO2': 'itcz-slab-4xCO2_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-4xCO2-m40': 'itcz-slab-4xCO2-m40_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-4xCO2-m20': 'itcz-slab-4xCO2-m20_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-4xCO2-p20': 'itcz-slab-4xCO2-p20_zonal_mean_climo_years_11_to_40.nc',
                    'itcz-slab-4xCO2-p40': 'itcz-slab-4xCO2-p40_zonal_mean_climo_years_11_to_40.nc',
                    }
    elif model == 'CESM2':
        filenames = {'itcz-SST': 'itcz-SST-ctl.h1.zm_clim.nc',
                    'itcz-slab': 'itcz-slab-ctl.h1.zm_clim.nc',
                    'itcz-slab-m40': 'itcz-slab-m40.h1.zm_clim.nc',
                    'itcz-slab-m20': 'itcz-slab-m20.h1.zm_clim.nc',
                    'itcz-slab-p20': 'itcz-slab-p20.h1.zm_clim.nc',
                    'itcz-slab-p40': 'itcz-slab-p40.h1.zm_clim.nc',
                    'itcz-slab-4xCO2': 'itcz-slab-4xCO2-ctl.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-m40': 'itcz-slab-4xCO2-m40.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-m20': 'itcz-slab-4xCO2-m20.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-p20': 'itcz-slab-4xCO2-p20.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-p40': 'itcz-slab-4xCO2-p40.h1.zm_clim.nc',
                    }
    elif model == 'CESM2-QOBS':
        filenames = {'itcz-SST': 'itcz-SST.h0.zm_clim.nc',
                    'itcz-slab': 'itcz-slab.h1.zm_clim.nc',
                    'itcz-slab-m40': 'itcz-slab-m40.h1.zm_clim.nc',
                    'itcz-slab-m20': 'itcz-slab-m20.h1.zm_clim.nc',
                    'itcz-slab-p20': 'itcz-slab-p20.h1.zm_clim.nc',
                    'itcz-slab-p40': 'itcz-slab-p40.h1.zm_clim.nc',
                    'itcz-slab-4xCO2': 'itcz-slab-4xCO2.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-m40': 'itcz-slab-4xCO2-m40.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-m20': 'itcz-slab-4xCO2-m20.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-p20': 'itcz-slab-4xCO2-p20.h1.zm_clim.nc',
                    'itcz-slab-4xCO2-p40': 'itcz-slab-4xCO2-p40.h1.zm_clim.nc',
                    }
    elif model == 'GFDL-AM2':
        filenames = {'itcz-SST': 'itcz-SST_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab': 'itcz-slab_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-m40': 'itcz-slab-m40_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-m20': 'itcz-slab-m20_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-p20': 'itcz-slab-p20_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-p40': 'itcz-slab-p40_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-4xCO2': 'itcz-slab-4xCO2_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-4xCO2-m40': 'itcz-slab-4xCO2-m40_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-4xCO2-m20': 'itcz-slab-4xCO2-m20_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-4xCO2-p20': 'itcz-slab-4xCO2-p20_1987-2016.atmos_month_zm_clim.nc',
                    'itcz-slab-4xCO2-p40': 'itcz-slab-4xCO2-p40_1987-2016.atmos_month_zm_clim.nc',
                    }
    elif model == 'Isca':
        filenames = {'itcz-SST': 'itcz-mip_zm_clim.nc',
                    'itcz-slab': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-m40': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-m20': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-p20': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-p40': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-4xCO2': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-4xCO2-m40': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-4xCO2-m20': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-4xCO2-p20': 'itcz-mip_zm_clim.nc',
                    'itcz-slab-4xCO2-p40': 'itcz-mip_zm_clim.nc',
                    }
    else:
        print("Warning: filenames are not defined for %s. Returning empty dict." % model)
        filenames = {}
        
    return filenames
    
    
def define_variable_names():
    """ Create dict to account for differing variable names in different models """
    
    var_names = {}
    
    # key is generalized variable name
    # item is variable name used in particular model
    var_names['CESM1'] = {'plev': 'lev',
                          'lat': 'lat',
                          'tas': 'T2',
                          'ts': 'TS',
                          'ta': 'T',
                          'zg': 'Z3',
                          'us': 'U10',
                          'ua': 'U',
                          'va': 'V',
                          'wa': 'OMEGA',
                          'hus': 'Q',
                          'prc': 'PRECC',
                          'prl': 'PRECL',
                          'pr': 'pr',
                          'sfc_net': 'sfc_net',
                          'rsdt': 'FSDTOA',
                          'evspsbl': 'evspsbl',
                          'pme': 'pme',
                          'psi': 'psi',
                          'psi500': 'psi500',
                          'ua850': 'ua850',
                          'theta': 'theta',
                          'cldlow': 'CLDLOW',
                          'toa_net': 'toa_net',
                          'toa_net_cld': 'toa_net_cld',
                          'toa_lw': 'toa_lw',
                          'toa_sw': 'toa_sw',
                          'toa_net_clr': 'toa_net_clr',
                          'toa_lw_clr': 'toa_lw_clr',
                          'toa_sw_clr': 'toa_sw_clr',
                          'toa_lw_cld': 'toa_lw_cld',
                          'toa_sw_cld': 'toa_sw_cld',
                         }
    
    var_names['CESM2'] = var_names['CESM1']
    var_names['CESM2-QOBS'] = var_names['CESM1']
    
    # U10 not output for CESM2, so use UBOT instead (probably not actually the same)
    var_names['CESM2']['us'] = 'UBOT'
    var_names['CESM2-QOBS']['us'] = 'UBOT'
    
    var_names['GFDL-AM2'] = {'plev': 'pfull',
                             'lat': 'lat',
                             'tas': 't_ref',
                             'ts': 't_surf',
                             'ta': 'temp',
                             'zg': 'z_full',
                             'us': 'u_ref',
                             'ua': 'ucomp',
                             'va': 'vcomp',
                             'wa': 'omega',
                             'hus': 'sphum',
                             'prc': 'prec_conv',
                             'prl': 'prec_ls',
                             'pr': 'pr',
                             'sfc_net': 'sfc_net',
                             'rsdt': 'swdn_toa',
                             'evspsbl': 'evspsbl',
                             'pme': 'pme',
                             'psi': 'psi',
                             'psi500': 'psi500',
                             'ua850': 'ua850',
                             'theta': 'theta',
                             'cldlow': 'low_cld_amt',
                          'toa_net': 'toa_net',
                          'toa_net_cld': 'toa_net_cld',
                          'toa_lw': 'toa_lw',
                          'toa_sw': 'toa_sw',
                          'toa_net_clr': 'toa_net_clr',
                          'toa_lw_clr': 'toa_lw_clr',
                          'toa_sw_clr': 'toa_sw_clr',
                          'toa_lw_cld': 'toa_lw_cld',
                          'toa_sw_cld': 'toa_sw_cld',
                            }
    
    var_names['Isca'] = {'plev': 'pfull',
                          'lat': 'lat',
                          'tas': 'tas',
                          'ts': 'ts',
                          'ta': 'ta',
                          'zg': 'zg',
                          'us': 'uas',
                          'ua': 'ua',
                          'va': 'va',
                          'wa': 'wap',
                          'hus': 'hus',
                          'prc': 'prc',
                          'pr': 'pr',
                          'prl': 'prl',
                          'sfc_net': 'sfc_net',
                          'rsdt': 'rsdt',
                          'evspsbl': 'evspsbl',
                          'pme': 'pme',
                          'psi': 'psi',
                          'psi500': 'psi500',
                          'ua850': 'ua850',
                          'theta': 'theta',
                          'cldlow': 'cldlow',
                          'toa_net': 'toa_net',
                          'toa_net_cld': 'toa_net_cld',
                          'toa_lw': 'toa_lw',
                          'toa_sw': 'toa_sw',
                          'toa_net_clr': 'toa_net_clr',
                          'toa_lw_clr': 'toa_lw_clr',
                          'toa_sw_clr': 'toa_sw_clr',
                          'toa_lw_cld': 'toa_lw_cld',
                          'toa_sw_cld': 'toa_sw_cld',
                        }
    
    return var_names


def define_dimension_rename_dict():
    """ Create dict to rename dimension names consistently across models """
    
    dimension_rename_dict = {}
    
    # key is dimension name used in particular model
    # item is common dimension name
    dimension_rename_dict['CESM1'] = {'lev': 'plev', 'lat': 'lat', 'ilev': 'iplev'}
    dimension_rename_dict['CESM2'] = dimension_rename_dict['CESM1']
    dimension_rename_dict['CESM2-QOBS'] = {'lev': 'plev', 'lat': 'lat'}
    dimension_rename_dict['GFDL-AM2'] = {'pfull': 'plev', 'lat': 'lat', 'phalf': 'iplev'}
    dimension_rename_dict['Isca'] = {'pfull': 'plev', 'lat': 'lat'}
    
    return dimension_rename_dict

