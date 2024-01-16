import xarray as xr
import numpy as np
from scipy import integrate
from . import load
xr.set_options(keep_attrs=True);
    
def calc_derived_vars(model, ds):
    g = 9.807  # acceleration due to graviy (m/s^2)
    a = 6.371e6  # radius of Earth (m)
    rho_water = 1000.0  # density of water (kg/m^3)
    R = 287.0  # gas constant of air (J/K/kg)
    cp = 1004.0  # specific heat at constant pressure (J/K/kg)
    s_per_day = 86400
    
    LHV = 2.5e6  # latent heat of vaporization (J/kg)
    LHF = 3.34e5  # latent heat of fusion (J/kg)
    snow_flux_conv = rho_water * LHF  # m/s to W/m2
    evap_flux_conv = rho_water * LHV  # m/s to W/m2
    
    pr_conv = {"GFDL-AM2": s_per_day,
               "CESM1": rho_water * s_per_day, 
               "CESM2": rho_water * s_per_day, 
               "CESM2-QOBS": rho_water * s_per_day}
    
    vn = load.define_variable_names()
    vn = vn[model]
    
    # calculate mass streamfunction
    prefactor = 2 * np.pi * a / g * np.cos(np.deg2rad(ds["lat"].values))
    ds["psi"] = xr.zeros_like(ds[vn["va"]])

    if ds["plev"].units in ["mb", "hPa", "level"]:
        plev_unit_conv = 100.0
        p0 = 1000.0
    else:
        plev_unit_conv = 1.0
        p0 = 100000.0

    for j in range(len(ds["lat"])):
        ds["psi"][:, j] = prefactor[j] * integrate.cumtrapz(
            ds[vn["va"]][:, j], ds["plev"] * plev_unit_conv, initial=0
        )

    ds["psi"].attrs["units"] = "kg/s"
    ds["psi"].attrs["long_name"] = "mass streamfunction"
        
    ds["psi500"] = ds["psi"].sel(plev=50000/plev_unit_conv, method="nearest")
    ds["psi500"].attrs["units"] = "kg/s"
    ds["psi500"].attrs["long_name"] = "500hPa mass streamfunction"

    # compute potential temperature
    ds["theta"] = ds[vn["ta"]] * (p0 / ds["plev"]) ** (R / cp)
    
    # compute MSE
    ds["MSE"] = LHV * ds[vn["hus"]] + g * ds[vn["zg"]] + cp * ds[vn["ta"]]
    ds["MSE"].attrs["units"] = "J/kg"
    ds["MSE"].attrs["long_name"] = "Moist static energy"

    ds["MSE_K"] = ds["MSE"] / cp
    ds["MSE_K"].attrs["units"] = "K"
    ds["MSE_K"].attrs["long_name"] = "MSE divided by cp"
    
    #ds["tas"].attrs["units"] = "K"
    ds[vn["ts"]].attrs["units"] = "K"
    
    
    # select certain variables at certain pressure levels
    for var in ["ua", "wa"]:
        for lev in [850, 500, 250]:
            ds["%s%s" % (var, lev)] = ds[vn[var]].sel(plev=100*lev/plev_unit_conv, method="nearest")
            try:
                ds["%s%s" % (var, lev)].attrs["units"] = ds[vn[var]].attrs["units"]
                ds["%s%s" % (var, lev)].attrs["long_name"] = "%shPa %s" % (lev, ds[vn[var]].attrs["long_name"])
            except:
                ds["%s%s" % (var, lev)].attrs["units"] = "n/a"
                ds["%s%s" % (var, lev)].attrs["units"] = "n/a"
    
    # compute total precipitation and P-E in mm/day
    if model in ["CESM1"]:
        ds["pr"] = (ds[vn["prc"]] + ds[vn["prl"]]) * pr_conv[model]
        ds["evspsbl"] = ds["LHFLX"] / evap_flux_conv * pr_conv[model]
    elif model in ["GFDL-AM2"]:
        ds["pr"] = ds["precip"] * pr_conv[model]
        ds["evspsbl"] = ds["evap"] * pr_conv[model]
    elif model in ["CESM2", "CESM2-QOBS"]:
        ds["pr"] = ds["PRECT"] * pr_conv[model]
        ds["evspsbl"] = ds["LHFLX"] / evap_flux_conv * pr_conv[model]
#    elif model in ["Isca"]:
#        ds["pr"] = ds["pr"] 
#        ds["evspsbl"] = ds["evspsbl"]


    ds["pme"] = ds["pr"] - ds["evspsbl"]
    ds["pr"].attrs["long_name"] = "total precipitation"
    ds["pme"].attrs["long_name"] = "precip minus evap"
    ds["evspsbl"].attrs["long_name"] = "evaporation"
        
    for var in ["pr", "evspsbl", "pme"]:
        ds[var].attrs["units"] = "mm/day"

        
def calc_surf_TOA_fluxes(model, ds):
    LHV = 2.5e6  # latent heat of vaporization (J/kg)
    LHF = 3.34e5  # latent heat of fusion (J/kg)
    rho_water = 1000.0  # density of water (kg/m^3)
    snow_flux_conv = rho_water * LHF  # m/s to W/m2
    evap_flux_conv = rho_water * LHV  # m/s to W/m2
    
    vn = load.define_variable_names()
    vn = vn[model]
    
    # compute net surface and TOA fluxes
    if model == "GFDL-AM2":
        ds["sfc_net"] = (
            ds["swdn_sfc"]
            - ds["swup_sfc"]
            + ds["lwdn_sfc"]
            - ds["lwup_sfc"]
            - LHV * ds["evap"]
            - LHF * (ds["snow_conv"] + ds["snow_ls"])
            - ds["shflx"]
        )
        
        # calculate TOA fluxes for all-sky (cs="") and clear-sky (cs="_clr")
        for cs in ["", "_clr"]:
            ds["toa_lw" + cs] = - ds["olr" + cs]
            ds["toa_sw" + cs] = ds["swdn_toa" + cs] - ds["swup_toa" + cs]
            ds["toa_net" + cs] = ds["toa_lw" + cs] + ds["toa_sw" + cs]
    elif model in ["CESM1", "CESM2", "CESM2-QOBS"]:
        ds["sfc_net"] = (
            ds["FSNS"]
            - ds["FLNS"]
            - ds["SHFLX"]
            - ds["LHFLX"]
            - snow_flux_conv * ds["PRECSC"]
            - snow_flux_conv * ds["PRECSL"]
        )
        
        for cs in ["", "C"]:
            if cs == "C":
                cs_output = "_clr"
            else:
                cs_output = ""
                
            ds["toa_lw" + cs_output] = - ds["FLUT" + cs]
            ds["toa_sw" + cs_output] = ds["FSNT" + cs]
            ds["toa_net" + cs_output] = ds["toa_lw" + cs_output] + ds["toa_sw" + cs_output]
    
    elif model == "Isca": 
        ds["sfc_net"] = (
            ds["rsds"] - ds["rsus"]
            + ds["rlds"] - ds["rlus"]
            - ds["hfss"] # global mean hfss is an order of magnitude different from CESM2
            - ds["hfls"] 
            # isca has no snow
   #         - snow_flux_conv * ds["PRECSC"] 
   #         - snow_flux_conv * ds["PRECSL"]
        )
        cs = ""
        cs_output = ""                
#        np.nan
        ds["toa_lw" + cs_output] = - ds["rlut"]
        ds["toa_sw" + cs_output] = ds["rsdt"] - ds["rsut"]
        ds["toa_net" + cs_output] = ds["toa_lw" + cs_output] + ds["toa_sw" + cs_output]
        cs_output = "_clr"  
        # isca has no clear sky radiative fluxes
        ds["toa_lw" + cs_output] = xr.DataArray(np.full(ds["toa_lw"].shape, np.nan),dims=["lat"])
        ds["toa_sw" + cs_output] = xr.DataArray(np.full(ds["toa_lw"].shape, np.nan),dims=["lat"])
        ds["toa_net" + cs_output] = xr.DataArray(np.full(ds["toa_lw"].shape, np.nan),dims=["lat"])
    
    
    # compute net energy input
    ds["NEI"] = ds["toa_net"] - ds["sfc_net"]
    
    # compute cloud net and SW toa
    for var in ["toa_net", "toa_lw", "toa_sw"]:
        ds[var + "_cld"] = ds[var] - ds[var + "_clr"]
    
    # define variable long names
    ds["sfc_net"].attrs["long_name"] = "Net surface flux (down=positive)"
    ds["toa_net"].attrs["long_name"] = "Net TOA flux (down=positive)"
    ds["NEI"].attrs["long_name"] = "Net energy input to atmosphere)"
    
    # define units
    for var in ["sfc_net", "toa_net", "toa_net_clr", "toa_net_cld", "toa_sw", "toa_sw_clr", "toa_sw_cld", "toa_lw", "toa_lw_clr", "toa_lw_cld", "NEI"]:
        ds[var].attrs["units"] = "W/m2"

    # compute downward solar flux at TOA for CESM2
    if model in ["CESM2", "CESM2-QOBS"]:
        ds["FSDTOA"] = ds["FSNTOA"] + ds["FSUTOA"]
        ds["FSDTOA"].attrs["units"] = "W/m2"
        ds["FSDTOA"].attrs["long_name"] = "Downwelling solar flux at top of atmosphere"
        
        
def calc_metrics(data, model_list, experiment_list):
    R_EARTH = 6.371e6  # radius of earth [m]
    G = 9.81  # acceleration due to gravity [m s^-2]
    TWO_PI_R2 = 2 * np.pi * R_EARTH ** 2
    vn = load.define_variable_names()

    metric_list = [
        "Hadley-psi500-SH",
        "ITCZ-psi500-SH",
        "ITCZ-psi500-zero-trop",
        "ITCZ-psi500-NH",
        "ITCZ-psimax-SH",
        "ITCZ-psimax-NH",
        "ITCZ-psimean-SH",
        "ITCZ-psimean-NH",
        "ITCZ-pme-NH",
        "ITCZ-pme-SH",
        "ITCZ-p4-NH",
        "ITCZ-p4-SH",
        "ITCZ-w500-SH",
        "ITCZ-w500-NH",
        "ITCZ-w250-SH",
        "ITCZ-w250-NH",
        "Hadley-psi500-NH",
        "Hadley-psimean-SH",
        "Hadley-psimean-NH",
        "psimean-minval-SH",
        "psimean-maxval-NH",
        "psi500-minval-SH",
        "psi500-maxval-NH",
        "jetlat-ua850-SH",
        "jetlat-ua850-NH",
        "ts-gm",
        "pr-gm",
        "toa_net-gm",
        "toa_lw-gm",
        "toa_sw-gm",
        "toa_net_cld-gm",
        "toa_lw_cld-gm",
        "toa_sw_cld-gm",
        "toa_net_clr-gm",
        "toa_lw_clr-gm",
        "toa_sw_clr-gm",
    ]
    
    metric_units = {
        "Hadley-psi500-SH": "deg lat",
        "ITCZ-psi500-SH": "deg lat",
        "ITCZ-psi500-zero-trop": "deg lat",
        "ITCZ-psi500-NH": "deg lat",
        "ITCZ-psimax-SH": "deg lat",
        "ITCZ-psimax-NH": "deg lat",
        "ITCZ-psimean-SH": "deg lat",
        "ITCZ-psimean-NH": "deg lat",
        "ITCZ-pme-SH": "deg lat",
        "ITCZ-pme-NH": "deg lat",
        "ITCZ-p4-SH": "deg lat",
        "ITCZ-p4-NH": "deg lat",
        "ITCZ-w500-SH": "deg lat",
        "ITCZ-w500-NH": "deg lat",
        "ITCZ-w250-SH": "deg lat",
        "ITCZ-w250-NH": "deg lat",
        "Hadley-psi500-NH": "deg lat",
        "Hadley-psimean-SH": "deg lat",
        "Hadley-psimean-NH": "deg lat",
        "psimean-minval-SH": "kg s^-1",
        "psimean-maxval-NH": "kg s^-1",
        "psi500-minval-SH": "kg s^-1",
        "psi500-maxval-NH": "kg s^-1",
        "ts-gm": "K",
        "pr-gm": "mm day^-1",
        "jetlat-ua850-SH": "deg lat",
        "jetlat-ua850-NH": "deg lat",
        "toa_net-gm": "W m^-2",
        "toa_lw-gm": "W m^-2",
        "toa_sw-gm": "W m^-2",
        "toa_net_cld-gm": "W m^-2",
        "toa_lw_cld-gm": "W m^-2",
        "toa_sw_cld-gm": "W m^-2",
        "toa_net_clr-gm": "W m^-2",
        "toa_lw_clr-gm": "W m^-2",
        "toa_sw_clr-gm": "W m^-2",
    }
    
    metric_short_names = {x:x for x in metric_list}
 
    
    def sd(lat):
        return np.sin(np.deg2rad(lat))


    # initialize dataset to store circulation diagnostics (initialize with nan)
    ds_metrics = xr.Dataset(coords={"model": model_list, "experiment": experiment_list})
    for metric_name in metric_list:
        ds_metrics[metric_name] = xr.DataArray(
            np.full((len(experiment_list), len(model_list)), np.nan),
            dims=["experiment", "model"],
        )
        ds_metrics[metric_name].attrs["units"] = metric_units[metric_name]
        ds_metrics[metric_name].attrs["short_name"] = metric_short_names[metric_name]
        
    # compute diagnostics for each model and experiment
    for model in model_list:
        for exp, ds in data[model].items():
            circ = calc_streamfunction_metrics(ds.lat.values, ds.psi500.values)
            data_ua850_NH = ds.ua850.where(ds.lat>0, drop=True)
            data_ua850_SH = ds.ua850.where(ds.lat<0, drop=True)
            jetlat_NH = calc_jet_lat(data_ua850_NH.lat.values, data_ua850_NH.values)
            jetlat_SH = calc_jet_lat(data_ua850_SH.lat.values, data_ua850_SH.values)
            itcz = calc_itcz_pme_metrics(ds.lat.values, ds.pme.values)
            itcz_pr10 = calc_itcz_pme_metrics(ds.lat.values, ds.pr.values - 4.)
            p_psi_max = ds.psi.where(ds.psi == ds.psi.max(), drop=True).squeeze().plev.values.item(0)
            circ_max = calc_streamfunction_metrics(ds.lat.values, ds.psi.sel(plev=p_psi_max).values)
            itcz_w500 = calc_itcz_pme_metrics(ds.lat.values, -1 * ds.wa500.values)
            itcz_w250 = calc_itcz_pme_metrics(ds.lat.values, -1 * ds.wa250.values)

            plev_index_below = np.argwhere(np.array(ds.plev==ds.plev.sel(plev=slice(300,700))[-1])==True)[0][0] + 2
            plev_index_first = np.argwhere(np.array(ds.plev==ds.plev.sel(plev=slice(300,700))[0])==True)[0][0]
            try:
                weights = ds.iplev[plev_index_first:plev_index_below].diff("iplev")
            except: 
                weights = ds.plev[plev_index_first:plev_index_below].diff("plev")
            psi_300_700_mean = np.average(np.array(ds.psi.sel(plev=slice(300,700))),axis=ds.psi.dims.index('plev'),weights=weights)
            circ_mean = calc_streamfunction_metrics(ds.lat.values, psi_300_700_mean)

            for metric_name in metric_list:
                if metric_name == "Hadley-psi500-SH":
                    metric_value = circ[0]
                elif metric_name == "ITCZ-psi500-SH":
                    metric_value = circ[1]
                elif metric_name == "ITCZ-psi500-zero-trop":
                    metric_value = circ[2]
                elif metric_name == "ITCZ-psi500-NH":
                    metric_value = circ[3]
                elif metric_name == "Hadley-psi500-NH":
                    metric_value = circ[4]
                elif metric_name == "psi500-minval-SH":
                    metric_value = circ[5]
                elif metric_name == "psi500-maxval-NH":
                    metric_value = circ[6]
                elif metric_name == "jetlat-ua850-SH":
                    metric_value = jetlat_SH
                elif metric_name == "jetlat-ua850-NH":
                    metric_value = jetlat_NH
                elif metric_name == "ITCZ-pme-NH":
                    metric_value = itcz[0]
                elif metric_name == "ITCZ-pme-SH":
                    metric_value = itcz[1]
                elif metric_name == "ITCZ-w500-NH":
                    metric_value = itcz_w500[0]
                elif metric_name == "ITCZ-w500-SH":
                    metric_value = itcz_w500[1]
                elif metric_name == "ITCZ-w250-NH":
                    metric_value = itcz_w250[0]
                elif metric_name == "ITCZ-w250-SH":
                    metric_value = itcz_w250[1]
                elif metric_name == "ITCZ-p4-NH":
                    metric_value = itcz_pr10[0]
                elif metric_name == "ITCZ-p4-SH":
                    metric_value = itcz_pr10[1]
                elif metric_name == "ITCZ-psimax-NH":
                    metric_value = circ_max[3]
                elif metric_name == "ITCZ-psimax-SH":
                    metric_value = circ_max[1]
                elif metric_name == "ITCZ-psimean-NH":
                    metric_value = circ_mean[3]
                elif metric_name == "ITCZ-psimean-SH":
                    metric_value = circ_mean[1]
                elif metric_name == "psimean-minval-SH":
                    metric_value = circ_mean[5]
                elif metric_name == "psimean-maxval-NH":
                    metric_value = circ_mean[6]
                elif metric_name == "Hadley-psimean-NH":
                    metric_value = circ_mean[0]
                elif metric_name == "Hadley-psimean-NH":
                    metric_value = circ_mean[4]
                elif metric_name[-2:] == "gm":
                    variable = metric_name[:-3]
                    metric_value = calc_global_mean(ds[vn[model][variable]])
                    
                ds_metrics[metric_name].loc[dict(model=model, experiment=exp)] = metric_value
  
    # compute some additional circulation metrics based on what's already computed
    ds_metrics["Hadley-psi500-extent"] = ds_metrics["Hadley-psi500-NH"] - ds_metrics["Hadley-psi500-SH"]
    ds_metrics["Hadley-psimean-extent"] = ds_metrics["Hadley-psimean-NH"] - ds_metrics["Hadley-psimean-SH"]
    ds_metrics["Hadley-psi500-extent-half"] = 0.5 * ds_metrics["Hadley-psi500-extent"] 
    ds_metrics["ITCZ-psi500-extent"] = ds_metrics["ITCZ-psi500-NH"] - ds_metrics["ITCZ-psi500-SH"]
    ds_metrics["ITCZ-psimax-extent"] = ds_metrics["ITCZ-psimax-NH"] - ds_metrics["ITCZ-psimax-SH"]
    ds_metrics["ITCZ-psimean-extent"] = ds_metrics["ITCZ-psimean-NH"] - ds_metrics["ITCZ-psimean-SH"]
    ds_metrics["ITCZ-w500-extent"] = ds_metrics["ITCZ-w500-NH"] - ds_metrics["ITCZ-w500-SH"]
    ds_metrics["ITCZ-w250-extent"] = ds_metrics["ITCZ-w250-NH"] - ds_metrics["ITCZ-w250-SH"]
    ds_metrics["ITCZ-pme-extent"] = ds_metrics["ITCZ-pme-NH"] - ds_metrics["ITCZ-pme-SH"]
    ds_metrics["ITCZ-p4-extent"] = ds_metrics["ITCZ-p4-NH"] - ds_metrics["ITCZ-p4-SH"]
    ds_metrics["ITCZ-psimean-area"] = TWO_PI_R2 * (sd(ds_metrics["ITCZ-psimean-NH"]) - sd(ds_metrics["ITCZ-psimean-SH"]))
    ds_metrics["ITCZ-psi500-area"] = TWO_PI_R2 * (sd(ds_metrics["ITCZ-psi500-NH"]) - sd(ds_metrics["ITCZ-psi500-SH"]))
    ds_metrics["Hadley-psi500-area"] = TWO_PI_R2 * (sd(ds_metrics["Hadley-psi500-NH"]) - sd(ds_metrics["Hadley-psi500-SH"]))
    ds_metrics["descent-psi500-area"] = ds_metrics["Hadley-psi500-area"] - ds_metrics["ITCZ-psi500-area"]
    ds_metrics["descent-psi500-extent"] = ds_metrics["Hadley-psi500-extent"] - ds_metrics["ITCZ-psi500-extent"]
    ds_metrics["ITCZ-psi500-strength"] = (
        - G
        * (ds_metrics["psi500-maxval-NH"] - ds_metrics["psi500-minval-SH"])
        / ds_metrics["ITCZ-psi500-area"]
    )
    ds_metrics["descent-psi500-strength"] = (
        - G
        * (ds_metrics["psi500-minval-SH"] - ds_metrics["psi500-maxval-NH"])
        / ds_metrics["descent-psi500-area"]
    )
    ds_metrics["upwardmassflux-psi500"] = -1e-9 / G * ds_metrics["ITCZ-psi500-strength"] * ds_metrics["ITCZ-psi500-area"]
    ds_metrics["downwardmassflux-psi500"] = 1e-9 / G * ds_metrics["descent-psi500-strength"] * ds_metrics["descent-psi500-area"]
    ds_metrics["ITCZ-psimean-strength"] = (
        - G
        * (ds_metrics["psimean-maxval-NH"] - ds_metrics["psimean-minval-SH"])
        / ds_metrics["ITCZ-psimean-area"]
    )
    ds_metrics["upwardmassflux-psimean"] = -1e-9 / G * ds_metrics["ITCZ-psimean-strength"] * ds_metrics["ITCZ-psimean-area"]
    ds_metrics["Hadley-psimean-area"] = TWO_PI_R2 * (sd(ds_metrics["Hadley-psimean-NH"]) - sd(ds_metrics["Hadley-psimean-SH"]))
    ds_metrics["descent-psimean-area"] = ds_metrics["Hadley-psimean-area"] - ds_metrics["ITCZ-psimean-area"]
    ds_metrics["descent-psimean-extent"] = ds_metrics["Hadley-psimean-extent"] - ds_metrics["ITCZ-psimean-extent"]
    ds_metrics["descent-psimean-strength"] = (
        - G
        * (ds_metrics["psimean-minval-SH"] - ds_metrics["psimean-maxval-NH"])
        / ds_metrics["descent-psimean-area"]
    )
    ds_metrics["downwardmassflux-psimean"] = 1e-9 / G * ds_metrics["descent-psimean-strength"] * ds_metrics["descent-psimean-area"]

    
    # average between SH and NH jet positions
    ds_metrics["jetlat-ua850"] = 0.5 * (ds_metrics["jetlat-ua850-NH"] - ds_metrics["jetlat-ua850-SH"])

    # define units for additional derived metrics
    ds_metrics["Hadley-psi500-extent"].attrs["units"] = "deg lat"
    ds_metrics["Hadley-psi500-extent-half"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-psi500-extent"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-w500-extent"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-psimax-extent"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-psimean-extent"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-pme-extent"].attrs["units"] = "deg lat"
    ds_metrics["ITCZ-p4-extent"].attrs["units"] = "deg lat"
    ds_metrics["Hadley-psi500-area"].attrs["units"] = "m^*2"
    ds_metrics["ITCZ-psimean-area"].attrs["units"] = "m^2"
    ds_metrics["ITCZ-psi500-area"].attrs["units"] = "m^2"
    ds_metrics["descent-psi500-area"].attrs["units"] = "m^2"
    ds_metrics["ITCZ-psi500-strength"].attrs["units"] = "Pa s^-1"
    ds_metrics["ITCZ-psimean-strength"].attrs["units"] = "Pa s^-1"
    ds_metrics["descent-psi500-strength"].attrs["units"] = "Pa s^-1"
    ds_metrics["descent-psimean-strength"].attrs["units"] = "Pa s^-1"
    ds_metrics["upwardmassflux-psi500"].attrs["units"] = "$10^9$ kg s^-1"
    ds_metrics["downwardmassflux-psi500"].attrs["units"] = "$10^9$ kg s^-1"
    ds_metrics["upwardmassflux-psimean"].attrs["units"] = "$10^9$ kg s^-1"
    ds_metrics["downwardmassflux-psimean"].attrs["units"] = "$10^9$ kg s^-1"
    ds_metrics["jetlat-ua850"].attrs["units"] = "deg lat"
    
    # define short names for additional derived metrics
    ds_metrics["Hadley-psi500-extent"].attrs["short_name"] = "Hadley-psi500-extent"
    ds_metrics["Hadley-psi500-extent-half"].attrs["short_name"] = "Hadley-psi500-extent-half"
    ds_metrics["ITCZ-psi500-extent"].attrs["short_name"] = "$\phi_{ITCZ}$"
    ds_metrics["ITCZ-psimax-extent"].attrs["short_name"] = "$\phi_{ITCZ,max}$"
    ds_metrics["ITCZ-psimean-extent"].attrs["short_name"] = "$\phi_{ITCZ,300-700}$"
    ds_metrics["ITCZ-w500-extent"].attrs["short_name"] = "$\phi_{ITCZ,w500}$"
    ds_metrics["ITCZ-w250-extent"].attrs["short_name"] = "$\phi_{ITCZ,w250}$"
    ds_metrics["ITCZ-pme-extent"].attrs["short_name"] = "$\phi_{ITCZ,p-e}$"
    ds_metrics["ITCZ-p4-extent"].attrs["short_name"] = "$\phi_{ITCZ,p>4}$"
    ds_metrics["Hadley-psi500-area"].attrs["short_name"] = "Hadley-psi500-area"
    ds_metrics["ITCZ-psimean-area"].attrs["short_name"] = "A$_{ITCZ}$"
    ds_metrics["ITCZ-psi500-area"].attrs["short_name"] = "A$_{ITCZ}$"
    ds_metrics["descent-psi500-area"].attrs["short_name"] = "descent-psi500-area"
    ds_metrics["ITCZ-psi500-strength"].attrs["short_name"] = "$\omega_{ITCZ}$"
    ds_metrics["ITCZ-psimean-strength"].attrs["short_name"] = "$\omega_{ITCZ}$"
    ds_metrics["descent-psi500-strength"].attrs["short_name"] = "$\omega_{descent}$"
    ds_metrics["descent-psimean-strength"].attrs["short_name"] = "$\omega_{descent}$"
    ds_metrics["upwardmassflux-psi500"].attrs["short_name"] = "$\Psi_{ITCZ}$"
    ds_metrics["downwardmassflux-psi500"].attrs["short_name"] = "$\Psi_{descent}$"
    ds_metrics["upwardmassflux-psimean"].attrs["short_name"] = "$\Psi_{ITCZ}$"
    ds_metrics["downwardmassflux-psimean"].attrs["short_name"] = "$\Psi_{descent}$"
    ds_metrics["jetlat-ua850"].attrs["short_name"] = "jetlat-ua850"
    
    return ds_metrics        


def calc_4xCO2_response(ds_metrics, exp_ctl_list):
    # compute 4xCO2 responses for all diagnostics
    for exp in exp_ctl_list:
        exp_spl = exp.split("-")

        if len(exp_spl) == 2:
            exp_4xCO2 = exp + "-4xCO2"
        else:
            exp_4xCO2 = exp_spl[0] + "-" + exp_spl[1] + "-4xCO2-" + exp_spl[2]

        # compute 4xCO2 responses and save as new experiment
        ds_response = ds_metrics.sel(experiment=exp_4xCO2) - ds_metrics.sel(experiment=exp)
        ds_response = ds_response.assign_coords(experiment=exp + "-4xCO2response")
        ds_response = ds_response.expand_dims("experiment")

        ds_metrics = xr.concat([ds_metrics, ds_response], dim="experiment")

    # compute values of 4xCO2 response relative to climatological value in control experiment
    # save with "-rel" suffix (for "relative")
    for var, da in ds_metrics.items():
        ds_metrics[var + "-rel"] = xr.full_like(da, np.nan)
        for exp_ctl in exp_ctl_list:
            exp_pert = exp_ctl + "-4xCO2response"
            ds_metrics[var + "-rel"].loc[dict(experiment=exp_pert)] = (
                100
                * ds_metrics[var].sel(experiment=exp_pert)
                / ds_metrics[var].sel(experiment=exp_ctl)
            )
            ds_metrics[var + "-rel"].attrs["units"] = "%"

    # compute values of relative response normalized by global mean temperature response
    # save with "-norm" suffix (for "normalized")
    for var, da in ds_metrics.items():
        if var[-4:] == "-rel":
            ds_metrics[var + "-norm"] = da / ds_metrics["ts-gm"]
            ds_metrics[var + "-norm"].attrs["units"] = da.attrs["units"] + " K^-1"
        
    return ds_metrics

    
def calc_streamfunction_metrics(lat, data_mastrfu, verbose=False):
    """
    Assume data_mastrfu has same shape as lat (i.e. is just for one level/time/longitude)
    
    This routine finds the latitude of the tropical extremum of the mass streamfunction
    in each hemisphere (i.e the ITCZ extent), the latitude of the zero-crossing 
    between these two extremum (i.e. ITCZ position), the latitudes of the zero-crossings 
    poleward of the tropical extremums (i.e. the Hadley cell extent) and the values of
    the mass streamfunction at their extrema in each hemisphere (i.e. each hemisphere's
    Hadley cell strength).
    
    It does make some assumptions about the structure of the mass streamfunction being
    reasonably Earth-like (specifically, the region of ascent must be within 35S and 35N)
    
    Parameters
    ----------
    lat: 1d array of latitude in degrees
    data_mastrfu: 1d array of mass streamfunction (same size as lat)
    """
    
    # find index of maximum/minimum
    mastrfu_argmax = np.argmax(data_mastrfu[np.logical_and(lat > -5, lat < 35)]) # must pick maximum in NH tropics
    mastrfu_argmin = np.argmin(data_mastrfu[np.logical_and(lat > -35, lat < 5)]) # must pick minimum in SH tropics

    lat_ind_5S = np.argmax(lat[lat < -5])
    lat_ind_35S = np.argmax(lat[lat < -35])

    mastrfu_argmax += lat_ind_5S + 1
    mastrfu_argmin += lat_ind_35S + 1

    # find five grid points surrounding the minimum and maximum
    lat_max = lat[mastrfu_argmax-2:mastrfu_argmax+3]
    lat_min = lat[mastrfu_argmin-2:mastrfu_argmin+3]
    mastrfu_max = data_mastrfu[mastrfu_argmax-2:mastrfu_argmax+3]
    mastrfu_min = data_mastrfu[mastrfu_argmin-2:mastrfu_argmin+3]
 
    # fit quadratic to data near extrema
    data_fit_max = np.polyfit(lat_max, mastrfu_max, deg=2)
    data_fit_min = np.polyfit(lat_min, mastrfu_min, deg=2)
    
    # find latitude of max/min
    mastrfu_max_lat = -data_fit_max[1] / (2 * data_fit_max[0])
    mastrfu_min_lat = -data_fit_min[1] / (2 * data_fit_min[0])
    
    # compute max/min
    mastrfu_max_val = np.polyval(data_fit_max, mastrfu_max_lat)
    mastrfu_min_val = np.polyval(data_fit_min, mastrfu_min_lat)

    assert mastrfu_min_lat < mastrfu_max_lat

    # find zero between these latitudes
    if data_mastrfu[mastrfu_argmax] < 0:
        mastrfu_zero_lat = mastrfu_max_lat
    elif data_mastrfu[mastrfu_argmin] > 0:
        mastrfu_zero_lat = mastrfu_min_lat
    else:
        data_mastrfu_ITCZ = data_mastrfu[mastrfu_argmin:mastrfu_argmax+1]
        lat_ITCZ = lat[mastrfu_argmin:mastrfu_argmax+1]

        lat_min_ind = np.argmin(abs(data_mastrfu_ITCZ))

        if data_mastrfu_ITCZ[lat_min_ind]>0:
            lat_ind0 = lat_min_ind-1
            lat_ind1 = lat_min_ind
        else:
            lat_ind0 = lat_min_ind
            lat_ind1 = lat_min_ind+1

        mastrfu_zero_lat = - (lat_ITCZ[lat_ind1] - lat_ITCZ[lat_ind0]) \
            * data_mastrfu_ITCZ[lat_ind0] / (data_mastrfu_ITCZ[lat_ind1] - data_mastrfu_ITCZ[lat_ind0]) \
            + lat_ITCZ[lat_ind0]

    # compute Hadley cell extent
    # find index of NH maximum and first negative streamfunction
    lat_NH = lat[lat > 0]
    mastrfu_NH = data_mastrfu[lat > 0]
    #print(mastrfu_NH)

    mastrfu_NH_argmax = np.argmax(mastrfu_NH[lat_NH < 30]) # must pick maximum in tropics

    # check if maximum is actually positive. if not, Hadley cell not defined.
    zero_ind = mastrfu_NH_argmax
    while mastrfu_NH[zero_ind + 1]>0 and zero_ind < len(lat_NH) - 2:
        zero_ind += 1
        
    hadley_extent_NH = -mastrfu_NH[zero_ind] / (mastrfu_NH[zero_ind + 1] - mastrfu_NH[zero_ind]) \
                    * (lat_NH[zero_ind + 1] - lat_NH[zero_ind]) + lat_NH[zero_ind]

    # find index of SH minimum and first positive streamfunction
    lat_SH = lat[lat < 0]
    mastrfu_SH = data_mastrfu[lat < 0]

    mastrfu_SH_argmin = np.argmin(mastrfu_SH)
    zero_ind = mastrfu_SH_argmin
    while mastrfu_SH[zero_ind - 1] < 0 and zero_ind > 1:
        zero_ind -= 1

    hadley_extent_SH = -mastrfu_SH[zero_ind - 1] / (mastrfu_SH[zero_ind] - mastrfu_SH[zero_ind - 1]) \
                   * (lat_SH[zero_ind] - lat_SH[zero_ind - 1]) + lat_SH[zero_ind - 1]

    return hadley_extent_SH, mastrfu_min_lat, mastrfu_zero_lat, mastrfu_max_lat, hadley_extent_NH, mastrfu_min_val, mastrfu_max_val


def calc_jet_lat(lat, data_u):
    # compute jet latitude by fitting quadratic to max wind point, and two points on either side
    # following Simpson and Polvani, 2016, GRL
    # assume data_u has same shape as lat (i.e. is just for one level/time/longitude)
    
    # find max u gridpoint
    u_max_ind = np.argmax(data_u)
    
    # restrict data and latitude to the max grid point and two points on either side
    # if maximum is at or near edge of grid (this shouldn't happen) just use grid
    # point latitude
    if u_max_ind == 0:
        u_max_lat = lat[u_max_ind]
    elif u_max_ind == 1:
        u_max_lat = lat[u_max_ind]
    elif u_max_ind == len(lat)-1:
        u_max_lat = lat[u_max_ind]
    elif u_max_ind == len(lat)-2:
        u_max_lat = lat[u_max_ind]
    else:
        lat = lat[u_max_ind-2:u_max_ind+3]
        data_u = data_u[u_max_ind-2:u_max_ind+3]
        
        # fit quadratic to data
        data_fit = np.polyfit(lat, data_u, deg=2)
        
        # find latitude of max
        u_max_lat = - data_fit[1] / (2 * data_fit[0])
    
    return u_max_lat


def calc_itcz_pme_metrics(lat, data_mastrfu, verbose=False):
    """
    Assume data_mastrfu has same shape as lat (i.e. is just for one level/time/longitude)
    
    This routine finds the latitude of the tropical zero-crossings of precipitation - evaporation
    in each hemisphere (i.e the ITCZ extent).
    
    It assumes the ITCZ is in the tropics centered around the equator.
    
    (Added by AGP, 24 Oct 2019)
    
    Parameters
    ----------
    lat: 1d array of latitude in degrees
    data_mastrfu: 1d array of p-e (same size as lat)
    """
    
    # find index of NH maximum and first negative P-E
    lat_NH = lat[lat > 0]
    mastrfu_NH = data_mastrfu[lat > 0]
    #print(mastrfu_NH)

    mastrfu_NH_argmax = np.argmax(mastrfu_NH[lat_NH < 30]) # must pick maximum in tropics

    # check if maximum is actually positive. if not, ITCZ width not defined.
    zero_ind = mastrfu_NH_argmax
    while mastrfu_NH[zero_ind + 1]>0 and zero_ind < len(lat_NH) - 2:
        zero_ind += 1

    ITCZ_extent_NH = -mastrfu_NH[zero_ind] / (mastrfu_NH[zero_ind + 1] - mastrfu_NH[zero_ind]) \
                    * (lat_NH[zero_ind + 1] - lat_NH[zero_ind]) + lat_NH[zero_ind]

    # find index of SH max and first negative P-E
    lat_SH = lat[lat < 0]
    mastrfu_SH = data_mastrfu[lat < 0]

    mastrfu_SH_argmax = np.argmax(mastrfu_SH)
    zero_ind = mastrfu_SH_argmax
    while mastrfu_SH[zero_ind - 1] > 0 and zero_ind > 1:
        zero_ind -= 1

    ITCZ_extent_SH = -mastrfu_SH[zero_ind - 1] / (mastrfu_SH[zero_ind] - mastrfu_SH[zero_ind - 1]) \
                   * (lat_SH[zero_ind] - lat_SH[zero_ind - 1]) + lat_SH[zero_ind - 1]

    return ITCZ_extent_NH, ITCZ_extent_SH



def calc_global_mean(da):
    lat = da.lat.values
    lat_rad = np.deg2rad(lat)
    
    area_weight = integrate.trapz(np.cos(lat_rad),lat_rad)
    global_integral = integrate.trapz(da.values * np.cos(lat_rad), lat_rad)
    
    return global_integral / area_weight