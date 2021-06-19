# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 09:22:04 2020

@author: Leo_Ji
"""
import numpy as np
import pandas as pd
import os

fdecimal = np.float32
idecimal = np.int32
HDF5_chunkSize = 1000000
min_eps = 1.e-10
min_axis = (0.05/4)**2
MBsize  = 1024**2

plot_lon_center = 0
save_blank = 8

inWin10 = True
################
anti_val = -777.
MeaningLess_val = {'missing': -999., 'none': -888., 'error': -999.}  # np.nan
tmp_missing = -950.

max_rainrate = 500.  # [mm h-1] 
raintypes = {'others': 3, 'convective': 2, 'stratiform': 1}
shallowR = {'NotShallow':0, 'Isolated': 1, 'NotIsolated': 2}
landocean = {'ocean': 0, 'land': 1, 'coast': 2, 'inLandLake': 3, 'SeaIce': 4, 'unknown': 9}  # from Ku 'status'
solidliquid= {'solid': 0, 'mixed': 1, 'liquid': 2, 'missing': 255}

pixel_area = {'TRMM_a': 4.3*4.3, 'TRMM': 5*5, 'GPM': 5*5}  # [km^2]
#fn_MJO_BSISO = 'phase_omi_index_1998-2020.csv'
fn_MJO_BSISO = 'phase_omi_index_1986-2019.hdf'
fn_adjoin_PFgrps = 'pf_grps_connected'
fn_Conv_adjoin_table = 'PF-Conv-adjoin-table'
fn_Ku_pfs = ['Ku_pf_GE500m2']

BM_dBZ = 35

DIR_datasets = ['ALL_pixels', 'Rainfall_pixels', 'dBZProfile_pixels', 'dBZHtop_pixels', 'TMI_KU_swath', 'TMI_swath', ] #'VIRS_KU_swath', 
DIR_final = 'Final'
DIR_Medata = 'InterMediate_data'
    
draw_ellipse = True

#BG_timestamp = pd.Timestamp(1993,1,1)
#BG_Totseconds = int((BG_timestamp-pd.to_datetime(0)).total_seconds())
#DT_missing = -BG_Totseconds-1000000
Years = {'TRMM': set(np.arange(1998,2015,1)), 'GPM': set(np.arange(2014,2021,1))}
Months = set(np.arange(1,13))

#DATA_source = ['TRMM2A25'] # [, 'TRMM1Z', 'TRMM1Z19', 'GPMKu' ]  # NOT used yet；['GPMDPR']
#DATA_source = ['TRMM1Z19']
#ana_time_rngs = ('20140401','20141231')

DATA_source = ['GPMKu']
ana_time_rngs = ('20140401','20140430')

### for windows
if inWin10:
    Disk = 'G:\\new'
    DIR_TRMM2A23 = 'TRMM_ku_2014\\2A23'
    DIR_TRMM2A25 = 'TRMM_ku_2014\\2A25'
#    DIR_GPMku = 'GPM_ku_2014'
    DIR_TRMM1Z19 = 'TRMM'
    DIR_GPMKu = 'DPR'
    DIR_MJO_BSISO = 'MJO'
    
    res_Disk = Disk
    op_res = '_results'
    op_data = 'compare_radar2'
    
    op_plot = os.path.join(res_Disk, op_res, "plot_BM")

    shp_file = 'D:\\static_data\\cartopy教程\\cn_shp\\Province_9\\Province_9.shp'
    terrain_file = 'D:\\static_data\\terrain\\terrain.hdf'
    
    stat_pect_Nsample = 2.e7
    
else:
### for linux
    Disk = '/alldata'
    DIR_TRMM2A23 = 'share/TRMM_ku_2014/2A23'
    DIR_TRMM2A25 = 'share/TRMM_ku_2014/2A25'
    DIR_TRMM1Z19 = 'data/GPM2020'
    DIR_GPMKu = 'data/GPM/DPR/new/2ADPR'
    DIR_MJO_BSISO = 'data/MJO'
    
#    res_Disk = '/home/leoji'
#    op_res = 'results'
#    op_data = 'TRMM'
    res_Disk = '/home/leoji'  #os.path.join(Disk, 'user','leoji')  #'/home/leoji'
    op_res = 'results'
    op_data = 'compare_radar2'   #'TRMM_global'  # 'GPM_global'
#    platform = op_data.split('_')[0]
    
    op_plot = '/home/leoji/results/plot_BM_TRMM2014'

    shp_file = '/home/leoji/static/Province_9/Province_9.shp'
    terrain_file = '/alldata/data/terrain.hdf'
    
    stat_pect_Nsample = 1.e8
    
_2A23_vars = {'status', 'shallowRain', 
              'HBB', 'BBwidth'}   # [m]
_2A25_vars = {'Year', 'Month', 'DayOfMonth', 'Hour', 'Minute', 'scanTime_sec',
              'Latitude', 'Longitude', 'dataQuality', #  'scLocalZenith',, 'scAlt'
              'correctZFactor', 'nearSurfRain', 'freezH', 'rainType'} #'nearSurfZ', 
_2A2X_var_adj = {100: set(['correctZFactor']),    ############# convert to [dBZ]                                               
                 1000: set(['freezH', 'HBB', 'BBwidth', 'scAlt']),    # convert to [km]
                  }

_1Z_vars = {'scan', 'ray', 'pr_dbz', 'pr.lon', 'pr.lat', 'pr.nearSurfRain', # 'pr.nearSurfZ', 
            'pr_2a23.raintype2A23', 'pr_2a23.hfreez2A23', 'pr_2a23.hbb2A23',
            'tmi.lathi', 'tmi.lonhi', 'tmi.h10', 'tmi.v10', 'tmi.h19', 'tmi.v19', 'tmi.v21', 'tmi.h37', 'tmi.v37', 'tmi.h85', 'tmi.v85',
            'colohi', 'tmi.pct37', 'tmi.pct85', 'tmi.surfprecip', 'tmi.surfaceflag',  # 'tmi.convprecip', 
#            'virs.ch4_all', 'virs.lat_all', 'virs.lon_all', 
            'virs.ch1', 'virs.ch2', 'virs.ch3', 'virs.ch4', 'virs.ch5'
            }
_1Z_vars_VS = {'pr_time', 'tmi_time'} #  'file_constants', 

_1Z_var_adj = {10: set(['pr.nearSurfRain', 'pr.Rain_2B31', 'pr.Precip_2B31',    # convert to [mm/h]
                       'tmi.scLat', 'tmi.scLon',    # convert to [degrees]
                       'tmi.rain', 'tmi.surfprecip', 'tmi.convprecip', 'tmi.precipwater',    # convert to [mm/h]
                       'tmi.pbrain', 'tmi.confidence',                           
                       'tmi.v10', 'tmi.h10', 'tmi.v19', 'tmi.h19', 'tmi.v21', 'tmi.v37', 'tmi.h37', 'tmi.v85', 'tmi.h85', 'tmi.pct37', 'tmi.pct85',    # # convert to [K]
                       'virs.ch1', 'virs.ch2', 'virs.ch3', 'virs.ch4', 'virs.ch5', 'virs.ch1_all', 'virs.ch2_all', 'virs.ch3_all', 'virs.ch4_all', 'virs.ch5_all',    # convert to [K]
                       'tmi.seasfct',    # convert to [K]
                       'tmi.windspeed']    # convert to [m/s]
                       ), 
               100: set(['pr_rain',    # convert to [mm/h]
                        'pr.lon', 'pr.lat', 'pr.lonpara', 'pr.latpara', 'tmi.lonhi', 'tmi.lathi', 'virs.lon_all', 'virs.lat_all',    # convert to [degrees]
                        'pr.nearSurfZ', 'pr.pia',    # convert to [dBZ]
                        'pr_lh', 'pr_q1mqr', 'pr_q2',    # convert to [K/h]
                        'tmi.rainwpath', 'tmi.icewpath', 'tmi.cldwpath']    ############# convert to [kg/m2] ?                        
                        ),
               1000: set(['pr_dbz', 'pr_2a23.hfreez2A23', 'pr_2a23.hbb2A23' ])    # convert to [dBZ]
                  }

_GPM_Ku_vars = {'NS/Longitude', 'NS/Latitude',
    'NS/ScanTime/Year','NS/ScanTime/Month','NS/ScanTime/DayOfMonth','NS/ScanTime/Hour','NS/ScanTime/Minute', #'NS/ScanTime/Second',
#    'NS/scanStatus/dataQuality',
    'NS/FLG/qualityFlag',
    'NS/PRE/elevation','NS/PRE/landSurfaceType',  #'NS/PRE/heightStormTop',  #'NS/PRE/zFactorMeasured',
    'NS/CSF/typePrecip', 'NS/CSF/heightBB',   #'NS/VER/heightZeroDeg',
#    'NS/DSD/phase','NS/SLV/paramDSD',
#    'NS/Experimental/precipRateESurface2',
    'NS/SLV/precipRateNearSurface','NS/SLV/zFactorCorrected',  #,, 'NS/SLV/precipRate'
    'NS/SLV/phaseNearSurface', 'NS/SLV/zFactorCorrectedNearSurface'
    }

_GPM_var_adj = {1000: set(['NS/CSF/heightBB', 'NS/PRE/elevation'])    #'NS/VER/heightZeroDeg',  convert to [km]
                  }

_1Z19_vars = ['PR__YEAR', 'PR__MONTH', 'PR__DAY', 'PR__HOUR',
            'PR__CORRECTZFACTOR', 'PR__LON', 'PR__LAT', 'PR__HBB', 'PR__NEARSURFZ', 
            'PR__NEARSURFRAIN', 'PR__RAINTYPE', 'PR__PHASE', #'COLO__RAY', 'COLO__SCAN',
            
            'TMI__SURFACEFLAG', 'COLO__HI', 
            
#            'TMI__YEAR', 'TMI__MONTH', 'TMI__DAY', 'TMI__HOUR',
#            'TMI__LATHI', 'TMI__LONHI', 'TMI__H10', 'TMI__V10', 'TMI__H19', 'TMI__V19', 'TMI__V21', 
#            'TMI__H37', 'TMI__V37', 'TMI__H85', 'TMI__V85', 'TMI__PCT10', 'TMI__PCT19', 'TMI__PCT37', 'TMI__PCT85',            
#            'TMI__SURFPRECIP', 'TMI__VALIDITY', # 'tmi.convprecip', 
#            'VIRS__CH1', 'VIRS__CH2', 'VIRS__CH3', 'VIRS__CH4', 'VIRS__CH5'
            ] 
_1Z19_var_adj = {1000: set(['PR__HBB']),    #'freezH', 'BBwidth',  convert to [km]
                  }

lev_dBZ = [20., 35.]  # [dBZ] , 30., 40.

zdim = {'TRMM': {'dH': 0.125, 'HGT': np.arange(0., 22., 0.125)[::-1].astype(fdecimal)},  # {'dH': 0.25, 'HGT': np.arange(0., 20., 0.25)[::-1].astype(fdecimal)},
        'GPM': {'dH': 0.125, 'HGT': np.arange(0., 22., 0.125)[::-1].astype(fdecimal)}}

Vchars = {'TMI_pct85':[100, 150, 175, 200, 225, 250],
          'TMI_pct37':[225, 250],
          'VIRS_ch4':[210, 235, 273]}



#######
gap = {'TRMM': 0.25, 'GPM': 0.5, 'TRMM_Lat': 1, 'GPM_Lat': 1, 'Hgt': 0.5}  # degree
limits = {'TRMM': [[-180, 180.000001], [-36.25, 36.250001]], 
          'GPM':  [[-180, 180.000001], [-66.5, 66.500001]],
          'TRMM_Lat': [[-36.5, 36.500001], [0,20.000001]],
          'GPM_Lat': [[-66.5, 66.500001], [0,20.000001]]}
shapes = {'TRMM': [int((limits['TRMM'][0][-1]-limits['TRMM'][0][0])/gap['TRMM']), int((limits['TRMM'][1][-1]-limits['TRMM'][1][0])/gap['TRMM'])], 
          'GPM':  [int((limits['GPM'][0][-1]-limits['GPM'][0][0])/gap['GPM']), int((limits['GPM'][1][-1]-limits['GPM'][1][0])/gap['GPM'])],
          'TRMM_Lat': [int((limits['TRMM_Lat'][0][-1]-limits['TRMM_Lat'][0][0])/gap['TRMM_Lat']), int((limits['TRMM_Lat'][1][-1]-limits['TRMM_Lat'][1][0])/gap['Hgt'])], 
          'GPM_Lat': [int((limits['GPM_Lat'][0][-1]-limits['GPM_Lat'][0][0])/gap['GPM_Lat']), int((limits['GPM_Lat'][1][-1]-limits['GPM_Lat'][1][0])/gap['Hgt'])]}






