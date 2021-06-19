# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:23:55 2021

@author: Leo.Ji
"""
import module_Parameters as param
from module_variables import Variable_calculations as Vcal

from scipy import stats
import numpy as np
import pandas as pd
import vaex as ve

from pyhdf.SD import SD, SDC     # for HDF4
from pyhdf.HDF import *
from pyhdf.VS import *

import h5py              # for HDF5

import os
import sys
import glob

import warnings
warnings.filterwarnings("ignore")


class Satellite_Ku:
    def __init__(self, mode='new'):
        if mode in ['new', 'append']:
            self.storage_mode = mode  # should be 'new': rewrite all data; 'append': append new data;
        else:
            print('"mode" should be one of ["new", "append"].')
            sys.exit(1)
                    
        self.YN = ''
        self.storage_checked = {}
                    
        self.__check_DIR__(param.op_plot, check_fn=False)
        
        self.data_end_index = pd.DataFrame()
        
        self.dataset = {'TRMM': {}, 'GPM': {}}
        self.MJO_BSISO = {}
        self.LO_ref = {}
        np.set_printoptions(suppress=True)
        
        self.float_accu = param.fdecimal
        self.int_accu = param.idecimal
        self.base_fn = 'Ku'
        self.data_subdirs = param.DIR_datasets
        return
    
    
    def __check_DIR__(self, DIR, init=False, check_fn=True):
        if not os.path.exists(DIR): os.makedirs(DIR)
        """
        if check_fn:
            res = glob.glob(os.path.join(DIR , '*.h5'))
            if (len(res) > 0) and (self.storage_mode == 'new'):            
                if self.YN.__len__() == 0:
                    print(f'[CAUTION]: There are HDF5 files in {DIR}. You should delete them or remove them first.')
                    self.YN = input("Are you sure to overlap all data exists?(Y/N)")
                if self.YN.lower() in ['n','no']:
                    sys.exit(1)
        """
        '''
        if init and (self.storage_mode == 'new'):
            res = glob.glob(os.path.join(DIR , '*.csv'))
            if len(res) > 0:
                [os.remove(fn) for fn in res]
        '''
        return
    
    
    def check_DIRs(self, platform):
        for subdir in [param.fn_adjoin_PFgrps, param.fn_Conv_adjoin_table]+self.data_subdirs+param.fn_Ku_pfs:
            if (subdir in ['TMI_KU_swath', 'VIRS_KU_swath', 'TMI_swath']) and (platform in ['GPM']): continue
            self.__check_DIR__(os.path.join(param.res_Disk, param.op_res, param.op_data, platform, subdir), init=True)
#            self.__check_DIR__(os.path.join(param.res_Disk, param.op_res, param.op_data, platform, subdir+'/month'))
#            if subdir in [param.fn_adjoin_PFgrps, param.fn_Conv_adjoin_table]: continue
#            self.__check_DIR__(os.path.join(param.res_Disk, param.op_res, param.op_data, platform, subdir+'/year'))
            
            self.storage_checked[subdir] = False
        return


    def read_HDF4(self, fn, Vflag='1Z19'):
        # http://pysclint.sourceforge.net/pyhdf/documentation.html           
#        print(fn)
        flag = -1  # default: BAD data
        # read scientific data part        
        SD_data, VS_data = {}, {}
        if '1Z09.20000103.12081.7.HDF' in fn:
            return flag, SD_data, VS_data
        sd = SD(fn, SDC.READ)   
        
#        for key in set(sd.datasets().keys()):
        if Vflag in ['1Z09']:
            QAvar = 'pr.nearSurfRain'
            varlist = param._1Z_vars - set(QAvar)
            
            ## QA step
            SD_data[QAvar] = sd.select(QAvar)[:]
            TFs = SD_data[QAvar] < 0            
            if (True in TFs) or (SD_data[QAvar].size > 0): flag = 0
            
            ## read VS data
            HDF_file = HDF(fn, HC.READ)
            vs = HDF_file.vstart()
            for vID in param._1Z_vars_VS:
                vd = vs.attach(vID)
                VS_data[vID] = pd.DataFrame(vd.read(nRec=vd.inquire()[0]), 
                                            columns=vd.inquire()[2])
                vd.detach()               # "close" the vdata
            vs.end()                  # terminate the vdata interface
            HDF_file.close()                 # close the HDF file
                
        elif Vflag in ['2A25']:
            QAvar = 'dataQuality'
            varlist = param._2A25_vars - set(QAvar)
            ## QA step
            SD_data[QAvar] = sd.select(QAvar)[:]
            TFs = SD_data[QAvar] == 0            
            if (True in TFs) or (SD_data[QAvar].size > 0): flag = 0
            
        elif Vflag in ['2A23']:
            flag = 0
            varlist = param._2A23_vars            
        elif Vflag in ['Clim_MB']:
            flag = 0
            varlist = sd.datasets().keys()

        if flag == 0:
            for key in varlist:
                SD_data[key] = sd.select(key)[:]
        sd.end()     
        return flag, SD_data, VS_data
    
    '''
    def read_terrain(self,):
        sd = SD(param.terrain_file, SDC.READ)
        self.LO_ref = sd.select('LOFLAG')[:]
        'LON', 'LAT'
        
        
        sd.end()
        return
    '''
    
    def read_HDF5(self, fn, Vflag='Ku'):
#        print(fn)
        flag, res, deal_TMI = -1, {}, False  # default: BAD data      
        
        MBsize = os.path.getsize(fn)/param.MBsize 
        if (Vflag not in ['1Z09', '1Z19']) and (MBsize < 50): return flag, res, deal_TMI  # file size is too small, treat as BAD data
        
        if Vflag in ['Ku']:
            QAvar = 'NS/FLG/qualityFlag'
            varlist = param._GPM_Ku_vars - set(QAvar)
        elif Vflag in ['1Z19']:
            QAvar = 'PR__flag'
            varlist = param._1Z19_vars
        elif Vflag in ['Ka']:
            pass
        elif Vflag in ['DPR']:
            pass
            
        try:
            data = h5py.File(fn,'r')
            ## first step: QA,
            ## We find that there are data with QAflag !=0 totally or partly. 
            ## So we should assign those ERROR data into -999, to make sure the rest calculations are meanfull.
            if Vflag in ['1Z19']:
                res['PR__NEARSURFRAIN'] = data['PR__NEARSURFRAIN'][:]
                res[QAvar] = (res['PR__NEARSURFRAIN']<0).astype(np.int8)
            else:
                res[QAvar] = data[QAvar][:]
            TFs = res[QAvar] == 0
            
            if (True in TFs) and (res[QAvar].size > 0):
                idxs = TFs == False
                for key in varlist:
                    res[key] = data[key][:]
                    if True in idxs:
                        if res[key].shape[0] != res[QAvar].shape[0]: continue
                        if res[key].ndim < 3:
                            res[key][idxs[:,0]] = param.MeaningLess_val['error']
                        elif res[key].ndim == 3:
                            res[key][idxs] = param.MeaningLess_val['error']
                
                flag = 0
#                deal_TMI = True
            data.close()
        except:
            if ('PR__' not in key) and ('SURFACEFLAG' not in key):
                flag, deal_TMI = 0, False
                keys = set(res.keys())
                for var in keys:
                    if ('PR__' in var) or ('SURFACEFLAG' in var): continue
                    res.pop(var)
                
            return flag, res, deal_TMI
            
        return flag, res, deal_TMI

    
    def Set_varlist(self, platform='TRMM'):
#        if platform in ['TRMM']:
        self.varlist_ALL_pixels = ['Longitude', 'Latitude', 'Year', 'Month', 'DayOfMonth', 'Hour', 'LocalHour', 'rainType', 'shallowRain', 'LandOcean', 'MJO_BSISO_Phase', 'MJO_BSISO_Magnitude']
        if 'shallowRain' not in self.dataset[platform].keys(): self.varlist_ALL_pixels.remove('shallowRain')
        
        self.varlist_Rain_pixels = list(self.dataset[platform].keys())
        [self.varlist_Rain_pixels.remove(var) for var in ['dBZ_profile','date','orbitID']]
        
        self.varlist_dBZ_Htop_pixels = ['Longitude', 'Latitude', 'Year', 'Month', 'LocalHour', 'rainType', 'LandOcean', 
                                        'MJO_BSISO_Phase', 'MJO_BSISO_Magnitude', 'ku_dBZ20_Htop', 'ku_dBZ35_Htop', 'ku_dBZ20_Npxl', 'ku_dBZ35_Npxl']
        
#        print(self.dataset[platform]['orbitID'], self.varlist_Rain_piexels.__len__())
        
        profile_levs = {f'_H{lev*1000:.0f}':self.dataset[platform]['dBZ_profile'][:,:,idx] 
                                       for idx, lev in enumerate(param.zdim[platform]['HGT'])}
        self.dataset[platform].update(profile_levs)
        self.varlist_dBZ_profile = self.varlist_ALL_pixels + list(profile_levs.keys())
        return 
    
    
    def get_datalist(self, ):
        for ds in param.DATA_source:
            if ds in ['TRMM1Z19']:
                platform = 'TRMM'
                datalist = {'1Z19': self.check_datalist(param.DIR_TRMM1Z19, suffix='HDF5', digDIR=True)}
            elif ds in ['GPMKu']:
                platform = 'GPM'
                datalist = {'Ku': self.check_datalist(param.DIR_GPMKu, suffix='HDF5', digDIR=True)}
            elif ds in ['TRMM2A25']:
                platform = 'TRMM'
                datalist = {'2A25': self.check_datalist(param.DIR_TRMM2A25, suffix='HDF'),
                            '2A23': self.check_datalist(param.DIR_TRMM2A23, suffix='HDF')}
            elif ds in ['TRMM1Z09']:
                platform = 'TRMM'
                datalist = {'1Z09': self.check_datalist(param.DIR_TRMM1Z09, suffix='HDF', digDIR=True)}
            #            elif ds in ['GPMDPR']:
#                platform = 'GPM'
#                datalist = {'Ku': self.check_datalist(param.DIR_GPMDPR, suffix='HDF5', walk=True)}
            else:
                platform, datalist = 'error', {}
        return platform, datalist
    
    
    def read_MJO_BSISO(self):
#        self.MJO_BSISO = pd.read_csv(os.path.join(param.Disk, param.DIR_MJO_BSISO, param.fn_MJO_BSISO))
        
        names = {'YEAR': ('Year', 'int32'), 'MONTH': ('Month', 'int32'), 'DAY': ('DayOfMonth', 'int32'), 'PHASE': ('Phase', 'int32'), 'MAGNITUDE': ('Magnitude', 'float32')}
        _, self.MJO_BSISO, _ = self.read_HDF4(os.path.join(param.Disk, param.DIR_MJO_BSISO, param.fn_MJO_BSISO), Vflag='Clim_MB')

        idx = np.where(self.MJO_BSISO['YEAR']==1998)[0][0]
        self.MJO_BSISO = pd.DataFrame({names[var][0]: np.squeeze(self.MJO_BSISO[var])[idx:].astype(names[var][1]) for var in self.MJO_BSISO.keys()})
        
        return 
    
    
    def Add_MJO_BSISO(self, platform):
        keys = ['Year', 'Month', 'DayOfMonth']
        base_df = pd.DataFrame({key: self.dataset[platform][key].ravel() for key in keys})
        res = base_df.merge(self.MJO_BSISO, on=keys, how='left').fillna(0)
        
        self.dataset[platform]['MJO_BSISO_Phase'] = res['Phase'].to_numpy().reshape(self.dataset[platform]['Year'].shape)
        self.dataset[platform]['MJO_BSISO_Magnitude'] = res['Magnitude'].to_numpy().reshape(self.dataset[platform]['Year'].shape)
        return
    
    
    def GPM_ku_preprocess(self, fn):
        flag, res, _ = self.read_HDF5(fn, Vflag='Ku')
        if flag == 0:
            res['date'] = fn.split('.')[-4].split('-')[0]
            res['orbitID'] = fn.split('.')[-3]
        else:
            print(f'[DATA ERROR:] {fn}')
        return flag, res
    
    
    def TRMM_ku_2A2X_preprocess(self, fn, list2A23):
        flag, res, _ = self.read_HDF4(fn, Vflag='2A25')
        if flag == 0:
            res['date'], res['orbitID'] = fn.split('.')[-4:-2]
            fn = fn.replace('2A25', '2A23')
            if fn in list2A23: res.update(self.read_HDF4(fn, Vflag='2A23')[1]) 
        else:
            print(f'[DATA ERROR:] {fn}')
        return flag, res, False
    
    
    def TRMM_1Z19_preprocess(self, fn):
        flag, res, deal_TMI = self.read_HDF5(fn, Vflag='1Z19')
#        print(fn)
        if flag == 0:
            res['date'] = fn.split('.')[-4]
            res['orbitID'] = fn.split('.')[-3]
        else:
            res = {}
            print(f'[DATA ERROR:] {fn}')
        return flag, res, deal_TMI
    
    
    def TRMM_1Z_preprocess(self, fn):
        flag, self.data_SD, self.data_VS = self.read_HDF4(fn, Vflag='1Z09')     
        if flag == 0:
            self.TRMM_1Z_adjust_data(fn)
        else:
            print(f'[DATA ERROR:] {fn}')
        return flag, self.data_SD
    
    
    def TRMM_1Z_adjust_data(self, fn):        
        tmp = np.where(self.data_SD['pr_2a23.raintype2A23']==168)
        if len(tmp[0]) > 0:
            self.data_SD['pr_2a23.raintype2A23'][tmp] = -88
        tmp = np.where(self.data_SD['pr_2a23.raintype2A23']==157)
        if len(tmp[0]) > 0:
            self.data_SD['pr_2a23.raintype2A23'][tmp] = -99
                             
        self.TRMM_1Z_pr_profile_filter()        
        ## build 3D profile field
        self.TRMM_1Z_pr_build3Dprofile()        
        
        [self.data_SD.pop(var) for var in ['scan', 'ray']]
                
        self.data_SD['Year'] = self.data_VS['pr_time']['pr.year'].values.astype(np.int32)
        self.data_SD['Month'] = self.data_VS['pr_time']['pr.month'].values.astype(np.int32)
        self.data_SD['DayOfMonth'] = self.data_VS['pr_time']['pr.day'].values.astype(np.int32)
        self.data_SD['Hour'] = self.data_VS['pr_time']['pr.hour'].values.astype(np.int32)
        self.data_SD['Minute'] = self.data_VS['pr_time']['pr.minute'].values.astype(np.int32)
        
        self.data_SD['Year_tmi'] = self.data_VS['tmi_time']['tmi.year'].values.astype(np.int32)
        self.data_SD['Month_tmi'] = self.data_VS['tmi_time']['tmi.month'].values.astype(np.int32)
        self.data_SD['DayOfMonth_tmi'] = self.data_VS['tmi_time']['tmi.day'].values.astype(np.int32)
        self.data_SD['Hour_tmi'] = self.data_VS['tmi_time']['tmi.hour'].values.astype(np.int32)
        self.data_SD['Minute_tmi'] = self.data_VS['tmi_time']['tmi.minute'].values.astype(np.int32)
        
        self.data_SD['date'], self.data_SD['orbitID'] = fn.split('.')[-4:-2]
        del self.data_VS
        return
    
    
    def TRMM_1Z_pr_build3Dprofile(self, ):
        tmpdata = np.full((self.data_SD['pr.nearSurfRain'].shape[0], self.data_SD['pr.nearSurfRain'].shape[1], self.data_SD['pr_dbz'].shape[0]), 0)
        tmpdata[self.data_SD['scan'], self.data_SD['ray'], :] = self.data_SD.pop('pr_dbz').T
        self.data_SD['pr_dbz'] = tmpdata        
        return 
    
    
    def TRMM_1Z_pr_profile_filter(self, ):
        # find and delete profiles, which is filled with zero or error.
        pfs = np.ma.array(self.data_SD['pr_dbz'], mask=self.data_SD['pr_dbz']<0)
        std = pfs.std(axis=0)
        rng = pfs.max(axis=0)-pfs.min(axis=0)
        cnt = np.count_nonzero(pfs, axis=0)
        
        idx = np.where(np.logical_and(np.logical_and(std<2., rng<10.), np.logical_or(cnt==0, cnt>65)))[0]
        if idx.size > 0:
            idx = idx[np.where(self.data_SD['pr.nearSurfRain'][self.data_SD['scan'][idx], self.data_SD['ray'][idx]]<=0)]
            for var in (set(['pr.nearSurfRain', 'pr.nearSurfZ', 'pr_2a23.hbb2A23']) & set(self.data_SD.keys())):
                self.data_SD[var][self.data_SD['scan'][idx], self.data_SD['ray'][idx]] = 0.           
            self.data_SD['pr_2a23.raintype2A23'][self.data_SD['scan'][idx], self.data_SD['ray'][idx]] = 300

            for var in ['scan', 'ray']:
                self.data_SD[var] = np.delete(self.data_SD[var], idx)
            self.data_SD['pr_dbz'] = np.delete(self.data_SD['pr_dbz'], idx, axis=1)                    
        return 
    
    
    def rename(self, flag, platform, deal_TMI=False):
        if flag in ['1Z19']:
            ## VIRS(K): ch1: 0.63 micron TB; ch2: 1.6 micron TB; ch3: 3.75 micron TB; ch4: 10.8 micron TB; ch5: 12 micron TB;
            ## TMI(K): [h/v] 10/ 19/ 21/ 37/ 85: 10/ 19/ 21/ 37/ 85 GhZ [Horizontal/Vertical] polarization TB;             
            rkeys = {'PR__CORRECTZFACTOR': 'dBZ_profile', 'PR__LON': 'Longitude', 'PR__LAT': 'Latitude', 
                     'PR__NEARSURFZ': 'nearSurfZ', 'PR__NEARSURFRAIN': 'nearSurfRain', 
                     'PR__HBB': 'HBB', #'PR__RAINTYPE': 'rainType', 'PR__PHASE': 'surfRainPhase',  #'pr_2a23.hfreez2A23': 'freezH', 
                     'TMI__PCT37': 'TMI_pct37', 'TMI__PCT85': 'TMI_pct85', 'TMI__PCT10': 'TMI_pct10', 'TMI__PCT19': 'TMI_pct19', 'TMI__SURFPRECIP': 'TMI_SurfPrecip',
                     'VIRS__CH1': 'VIRS_ch1', 'VIRS__CH2': 'VIRS_ch2', 'VIRS__CH3': 'VIRS_ch3', 'VIRS__CH4': 'VIRS_ch4', 'VIRS__CH5': 'VIRS_ch5',
                     
                     'PR__YEAR': 'Year', 'PR__MONTH': 'Month', 'PR__DAY': 'DayOfMonth', 'PR__HOUR': 'Hour', 
                     'TMI__SURFACEFLAG': 'TMI_LandOcean19', 'COLO__HI': 'colohi'}
                     
            if deal_TMI:         
                rkeys.update({'TMI__YEAR': 'Year_tmi', 'TMI__MONTH': 'Month_tmi', 'TMI__DAY': 'DayOfMonth_tmi', 'TMI__HOUR': 'Hour_tmi',
                     
                     'TMI__LATHI': 'TMI_Lat', 'TMI__LONHI': 'TMI_Lon', 'TMI__H10': 'TMI_H10', 'TMI__V10': 'TMI_V10', 'TMI__H19': 'TMI_H19', 'TMI__V19': 'TMI_V19', 
                     'TMI__V21': 'TMI_V21', 'TMI__H37': 'TMI_H37', 'TMI__V37': 'TMI_V37', 'TMI__H85': 'TMI_H85', 'TMI__V85': 'TMI_V85',                      
#                     'tmi.h10_all': 'TMI_H10_all', 'tmi.v10_all': 'TMI_V10_all', 'tmi.h19_all': 'TMI_H19_all', 'tmi.v19_all': 'TMI_V19_all', 
#                     'tmi.v21_all': 'TMI_V21_all', 'tmi.h37_all': 'TMI_H37_all', 'tmi.v37_all': 'TMI_V37_all', 'tmi.h85_all': 'TMI_H85_all', 'tmi.v85_all': 'TMI_V85_all', 
#                     'tmi.pct37_all': 'TMI_pct37_all', 'tmi.pct85_all': 'TMI_pct85_all', 'tmi.surfaceflag_all': 'TMI_LandOcean_all', 'tmi.surfprecip_all': 'TMI_SurfPrecip_all'                     
                     })
            
        elif flag in ['Ku']:
            rkeys = {'NS/Longitude': 'Longitude', 'NS/Latitude': 'Latitude',
                     'NS/ScanTime/Year': 'Year','NS/ScanTime/Month': 'Month','NS/ScanTime/DayOfMonth': 'DayOfMonth',
                     'NS/ScanTime/Hour': 'Hour','NS/ScanTime/Minute': 'Minute',
#                     'NS/FLG/qualityFlag',
                     'NS/PRE/elevation': 'Elevation',#'NS/PRE/landSurfaceType': 'LandOcean',  
#                     'NS/VER/heightZeroDeg': 'freezH',  #'NS/CSF/typePrecip': 'rainType',
                     'NS/SLV/zFactorCorrectedNearSurface': 'nearSurfZ',
                     'NS/CSF/heightBB': 'HBB',
#                     'NS/SLV/phaseNearSurface': 'surfRainPhase', 
                     'NS/SLV/precipRateNearSurface': 'nearSurfRain', 
                     'NS/SLV/zFactorCorrected': 'dBZ_profile'}
            
        elif flag in ['1Z09']:
            ## VIRS(K): ch1: 0.63 micron TB; ch2: 1.6 micron TB; ch3: 3.75 micron TB; ch4: 10.8 micron TB; ch5: 12 micron TB;
            ## TMI(K): [h/v] 10/ 19/ 21/ 37/ 85: 10/ 19/ 21/ 37/ 85 GhZ [Horizontal/Vertical] polarization TB;             
            rkeys = {'pr_dbz': 'dBZ_profile', 'pr.lon': 'Longitude', 'pr.lat': 'Latitude', 
                     'pr.nearSurfZ': 'nearSurfZ', 'pr.nearSurfRain': 'nearSurfRain', 
                     'pr_2a23.raintype2A23': 'rainType', 'pr_2a23.hfreez2A23': 'freezH', 'pr_2a23.hbb2A23': 'HBB',
                     'tmi.pct37': 'TMI_pct37', 'tmi.pct85': 'TMI_pct85', 'tmi.surfprecip': 'TMI_SurfPrecip',
                     'virs.ch1': 'VIRS_ch1', 'virs.ch2': 'VIRS_ch2', 'virs.ch3': 'VIRS_ch3', 'virs.ch4': 'VIRS_ch4', 'virs.ch5': 'VIRS_ch5',
                     
                     'tmi.surfaceflag': 'TMI_LandOcean',
                     'tmi.lathi': 'TMI_Lat', 'tmi.lonhi': 'TMI_Lon', 'tmi.h10': 'TMI_H10', 'tmi.v10': 'TMI_V10', 'tmi.h19': 'TMI_H19', 'tmi.v19': 'TMI_V19', 
                     'tmi.v21': 'TMI_V21', 'tmi.h37': 'TMI_H37', 'tmi.v37': 'TMI_V37', 'tmi.h85': 'TMI_H85', 'tmi.v85': 'TMI_V85',                      
#                     'tmi.h10_all': 'TMI_H10_all', 'tmi.v10_all': 'TMI_V10_all', 'tmi.h19_all': 'TMI_H19_all', 'tmi.v19_all': 'TMI_V19_all', 
#                     'tmi.v21_all': 'TMI_V21_all', 'tmi.h37_all': 'TMI_H37_all', 'tmi.v37_all': 'TMI_V37_all', 'tmi.h85_all': 'TMI_H85_all', 'tmi.v85_all': 'TMI_V85_all', 
#                     'tmi.pct37_all': 'TMI_pct37_all', 'tmi.pct85_all': 'TMI_pct85_all', 'tmi.surfaceflag_all': 'TMI_LandOcean_all', 'tmi.surfprecip_all': 'TMI_SurfPrecip_all'
                     }
        
        elif flag in ['2A25']:
            rkeys = {'correctZFactor': 'dBZ_profile'}
        
        for oldK, newK in rkeys.items():
            if oldK in self.dataset[platform].keys():
                self.dataset[platform][newK] = self.dataset[platform].pop(oldK)
        return
    
        
    def Manage_handler(self, ):
        platform, datalist = self.get_datalist()
        self.check_DIRs(platform)
        self.read_MJO_BSISO()
#        self.read_terrain()
        if platform in ['TRMM']:
            if '1Z19' in datalist.keys():
                for fn in datalist['1Z19']:
                    flag, res, deal_TMI = self.TRMM_1Z19_preprocess(fn)
                    if flag < 0: continue
                
                    self.dataset[platform].update(res)
                    self.Re_unit(platform, ratio_table=param._1Z19_var_adj)
                    self.rename('1Z19', platform, deal_TMI=deal_TMI)                                        
                    self.processes(fn, platform, deal_TMI=deal_TMI)
            elif '2A25' in datalist.keys():
                for fn in datalist['2A25']:
                    flag, res = self.TRMM_ku_2A2X_preprocess(fn, datalist['2A23'])
                    if flag < 0: continue
                
                    self.dataset[platform].update(res)
                    self.Re_unit(platform, ratio_table=param._2A2X_var_adj)
                    self.rename('2A25', platform)
                    self.processes(fn, platform)
            elif '1Z09' in datalist.keys():
                for fn in datalist['1Z09']:           
                    flag, res = self.TRMM_1Z_preprocess(fn)
                    if flag < 0: continue
                
                    self.dataset[platform].update(res)
                    del self.data_SD
                    self.Re_unit(platform, ratio_table=param._1Z_var_adj)
                    self.rename('1Z09', platform)                                        
                    self.processes(fn, platform, deal_TMI=True)
                
        elif platform in ['GPM']:
            if 'Ku' in datalist.keys():
                for fn in datalist['Ku']:
#                    fn = r'E:\Data\DPR\201507\2A.GPM.DPR.V8-20180723.20150711-S233124-E010355.007769.V06A.HDF5'  ## test
                    flag, res = self.GPM_ku_preprocess(fn)
                    if flag < 0: continue
                
                    self.dataset[platform].update(res)
                    self.Re_unit(platform, ratio_table=param._GPM_var_adj)
                    self.rename('Ku', platform)
                    self.processes(fn, platform)
        
#        self.data_end_index.to_csv('Pixel_data_record.csv')  # write
        return
    
    
    def processes(self, fn, platform, deal_TMI=False):
        self.Generate_variables(platform, deal_TMI=deal_TMI)
        QAflag = self.Quality_control(platform)
        if QAflag == 0:
            self.Mod_dtype(platform)
            
            self.Add_MJO_BSISO(platform)
            
            self.Set_varlist(platform)
            self.Tablelike_process(platform)  
            self.data_end_index.to_csv('Pixel_data_record.csv')
        else:
            print(f'[DATA ERROR:  QA] {fn}')
        return 
    
    
    def Tablelike_process(self, platform='TRMM'):
        date = self.dataset[platform]['date']  #, self.dataset[platform]['orbitID']
        ## ALL obs pixels
        subdir=self.data_subdirs[0]
        self.data_storage('_'.join([platform,self.base_fn, subdir, date]), platform, varlist=self.varlist_ALL_pixels, subdir=subdir)
        
        ## Rainfall pixels
        idxs = np.where(self.dataset[platform]['rainType']>0)
        subdir = self.data_subdirs[1]
        self.data_storage('_'.join([platform,self.base_fn, subdir, date]), platform, varlist=self.varlist_Rain_pixels, subset=idxs, subdir=subdir)
                        
        ## Profile pixels        
#        idxs = np.where(self.dataset[platform]['dBZ_profile'].max(axis=2)>0)
        self.dataset[platform].pop('dBZ_profile')
        subdir = self.data_subdirs[2]
        self.data_storage('_'.join([platform,self.base_fn, subdir, date]), platform, varlist=self.varlist_dBZ_profile, subset=idxs, subdir=subdir)
        
        # dBZHtop_pixels
        idxs = np.where(self.dataset[platform]['ku_dBZ20_Htop']>0)
        subdir = self.data_subdirs[3]
        self.data_storage('_'.join([platform,self.base_fn, subdir, date]), platform, varlist=self.varlist_dBZ_Htop_pixels, subset=idxs, subdir=subdir)
                
        self.dataset[platform].clear()
        return 
    
    
    def Quality_control(self, platform='TRMM'):
        dadict = self.dataset[platform]
        QAflag = -1
        if 'dataQuality' in dadict.keys():
            idxs = np.where(dadict['dataQuality']>0)[0]
            if idxs.size > 0:
                idxs = np.where(dadict['dataQuality']==0)[0]
                if idxs[0].size > 0:
                    QAflag = 0   # has good data partly
                    keys = set(dadict.keys()) - set(['orbitID', 'date', 'dataQuality'])
                    for key in keys:
                        dadict[key] = dadict[key][idxs]
            else:
                QAflag = 0   # good data
            dadict.pop('dataQuality')
        elif 'NS/FLG/qualityFlag' in dadict.keys():
            idxs = np.where(dadict['NS/FLG/qualityFlag']>0)[0]
            if idxs.size > 0:
                idxs = sorted(list(set(np.where(dadict['NS/FLG/qualityFlag']==0)[0])))
                if idxs[0].size > 0:
                    QAflag = 0   # has good data partly
                    keys = set(dadict.keys()) - set(['orbitID', 'date', 'NS/FLG/qualityFlag'])
                    for key in keys:
                        dadict[key] = dadict[key][idxs]
            else:
                QAflag = 0   # good data
            dadict.pop('NS/FLG/qualityFlag')
        elif 'PR__flag' in dadict.keys():
#            idxs = np.where(dadict['nearSurfRain']<0)
            idxs = np.where(dadict['PR__flag']>0)[0]
            if idxs.size > 0:
                idxs = np.setdiff1d(np.arange(dadict['PR__flag'].shape[0]), np.array(list(set(idxs)))) 
                if idxs[0].size > 0:
                    QAflag = 0   # has good data partly
                    keys = set(dadict.keys()) - set(['orbitID', 'date'])
                    for key in keys:
                        dadict[key] = dadict[key][idxs]
            else:
                QAflag = 0   # good data
            dadict.pop('PR__flag')
                
        if QAflag == 0:
            keys = set(dadict.keys()) & set(['nearSurfRain', 'TMI_SurfPrecip', 'dBZ_profile'])
            self.assign_nan(dadict, keys, has_thh=False)
#            for key in keys:
#                dadict[key][dadict[key]<0] = np.nan
            keys = set(dadict.keys()) & set(['HBB','freezH', 'TMI_pct10', 'TMI_pct19', 'TMI_pct37', 'TMI_pct85', 'VIRS_ch4'])
            self.assign_nan(dadict, keys, has_thh=True)
#            for key in keys:
#                dadict[key][dadict[key]<=0] = np.nan
        return QAflag
    
    
    def Re_unit(self, platform='TRMM', ratio_table={}):
        dadict = self.dataset[platform]
        for ratio in set(ratio_table.keys()):
            [self.sub_adj(dadict, var, ratio) for var in ratio_table[ratio] if var in dadict.keys()]            
        return 
    
    
    def Generate_variables(self, platform='TRMM', deal_TMI=False):
        dadict = self.dataset[platform]
        newvars = Vcal(dadict, platform=platform, deal_TMI=deal_TMI)
        dadict = newvars.variable_handler_A()
        
        if deal_TMI: 
            self.deal_TMI_dataset(platform)
        elif platform in ['TRMM']:
            dadict['LandOcean'] = dadict['LandOcean'].ravel()[dadict['colohi']]
            dadict.pop('colohi') 
        
        dadict = newvars.variable_handler_B()
        return 
    
    
    def Mod_dtype(self, platform='TRMM'):
        dadict = self.dataset[platform]
        keys = set(dadict.keys())
        for key in keys:
            if isinstance(dadict[key], str): continue
            
            if len(dadict[key].shape)==1:  # transform all variables into 2D shape
                dadict[key] = np.tile(dadict[key],(dadict['Latitude'].shape[1], 1)).T
                
                
            if dadict[key].dtype in ['float64', 'float16']:
                dadict[key] = dadict[key].astype(self.float_accu)
            if dadict[key].dtype in ['int64', 'int16', 'int8', 'uint8']:
                dadict[key] = dadict[key].astype(self.int_accu)
        return 
    
    
    def sub_adj(self, data, var, ratio):
        data[var] = (data[var] / ratio).astype(self.float_accu)
        return
    
    
    def generate_date_range(self, time_rngs=[],freq='M'):
        if time_rngs.__len__() != 2: time_rngs = param.ana_time_rngs
        res = set(pd.date_range(start=time_rngs[0], end=time_rngs[1], freq=freq).to_series() \
                            .apply(lambda x: x.strftime('%Y%m')).reset_index(drop=True))  #.to_list()
        return res
    
    
    def check_datalist(self, subdir, suffix='HDF', digDIR=False):  
        if digDIR:
            dalist = []
            dir_list = sorted(self.generate_date_range() & set(os.listdir(os.path.join(param.Disk, subdir))))
            for idir in dir_list:
                dalist.extend(sorted(glob.glob(os.path.join(param.Disk, subdir , idir, '*.'+suffix))))
        else:
            dalist = sorted(glob.glob(os.path.join(param.Disk, subdir , '*.'+suffix)))  
        return dalist
    
    
    def data_storage0(self, fn, platform, varlist=(), subset=(), subdir=''):
        dadict = self.dataset[platform]
        sfn = os.path.join(param.res_Disk, param.op_res, param.op_data, subdir, fn+'.h5')
        store_pixel = h5py.File(sfn, 'w')                         
        grp = store_pixel.create_group('data')
        
        if len(varlist)== 0: varlist = set(dadict.keys())
        for key in varlist:
            if len(subset) == 0:
                data = dadict[key].ravel()
            else:
                data = dadict[key][subset]
            
            grp.create_dataset(key, data.shape, data=data, compression="lzf") #, compression_opts=9
        store_pixel.close()
        return 
    
    
    def write_csv(self, fn, df):
        df.to_csv(fn)        
        return 
    
    
    def add_df_column(self, df, colname):
        if colname not in df.columns:
            df[colname] = np.zeros(13, dtype=np.int64)
        return df
    
    
    def data_storage(self, fn, platform, varlist=(), subset=(), subdir=''):
        dadict = self.dataset[platform]
        fDIR = os.path.join(param.res_Disk, param.op_res, param.op_data, platform, subdir)
        fn_month = os.path.join(fDIR, fn[:-2]+'.h5')  # 'month', 
#        fn_year = os.path.join(fDIR, 'year', fn[:-4]+'.h5')
    
        ###### check the file open mode and the start point of current dataset
        colname = fn.split('_')[-1][:4]+'_'+subdir                
        if (self.storage_mode in ['new']) and (not self.storage_checked[subdir]):
            self.data_end_index = self.add_df_column(self.data_end_index, colname=colname)            
            s0M = 0     #, s0Y = 0
            self.storage_checked[subdir] = True
            
        else:  # ['append']
            if self.storage_checked[subdir]:
                self.data_end_index = self.add_df_column(self.data_end_index, colname=colname) 
                s0M = self.data_end_index.loc[int(fn.split('_')[-1][4:6]), colname]
#                s0Y = self.data_end_index.loc[0, colname]
            else:
                datmp = ve.open(fn_month)
                s0M = datmp.shape[0]
                datmp.close()
#                datmp = ve.open(fn_year)
#                s0Y = datmp.shape[0]
#                datmp.close()
                self.storage_checked[subdir] = True
        
        #### write dataset
        loop = 0
        for sfn, s0 in zip([fn_month], [s0M]):  # [fn_month, fn_year], [s0M, s0Y]
            store_pixel = h5py.File(sfn, 'a')                   
            if 'data' in store_pixel.keys(): 
                grp = store_pixel['data']
            else:
                grp = store_pixel.create_group('data')
                    
            if len(subset) == 0:
                s1 = s0 + dadict[varlist[0]].shape[0]*dadict[varlist[0]].shape[1]
            else:
                s1 = s0 + subset[0].size                 
                
            for key in varlist:  
                if key in grp.keys():
                    datas = grp[key]
                else:
                    datas = grp.create_dataset(key, [param.HDF5_chunkSize], maxshape=[None], compression="lzf", chunks=True, dtype=dadict[key].dtype) #, compression_opts=9    
                    
                if len(subset) == 0:
                    datas.resize(s1, axis=0)
                    datas[s0:s1] = dadict[key].ravel()
                else:
                    datas.resize(s1, axis=0)
                    datas[s0:s1] = dadict[key][subset]
                    
            for key in ['orbitID']:
                if key in grp.keys():
                    datas = grp[key]
                else:
                    datas = grp.create_dataset(key, [param.HDF5_chunkSize], maxshape=[None], compression="lzf", chunks=True, dtype=np.int32)
                datas.resize(s1, axis=0)
                datas[s0:s1] = int(dadict['orbitID'])   # add orbit ID column
            store_pixel.close()
            
            ## record dataset length
            if loop == 0: 
                idx = int(fn.split('_')[-1][4:6])
            else:
                idx = 0
            self.data_end_index.loc[idx, colname] = s1
            loop += 1
        return 
    
    
    def data_load(self, fn='', varlist={}, getbase=True):
        res = {}
        
        data = h5py.File(fn, 'r')        # refer to writing method01
        for key in data.keys():
            if 'data' in key:
                res.update({'data': ve.from_dict({ikey.replace('/', ''): data[key].get(ikey)[:] 
                                                   for ikey in data[key].keys()}) })
            else:
                res.update({key.replace('/', ''): data.get(key)[:]})                
       
        data.close()
        return res
    
    
    def deal_TMI_dataset(self, platform):        
        ## change pct37 shape into pct85 shape
        self.TRMM_1Z_TMI_changeSize(platform)        
        self.TMI_QA(platform)
        ## assign tmi's, variables into pr/ku pixels
        self.TRMM_1Z_AssignTo_pr_pixels(platform)
        
        varlist = ['nearSurfRain', 'TMI_SurfPrecip', 'TMI_pct10', 'TMI_pct19', 'TMI_pct37', 'TMI_pct85', 
                   'TMI_H10', 'TMI_V10', 'TMI_H19', 'TMI_V19', 'TMI_V21', 'TMI_H37', 'TMI_V37', 'TMI_H85', 'TMI_V85',
                   'VIRS_ch1', 'VIRS_ch2', 'VIRS_ch3', 'VIRS_ch4', 'VIRS_ch5', 
                   'TMI_SurfPrecip_all', 'TMI_pct10_all', 'TMI_pct19_all', 'TMI_pct37_all', 'TMI_pct85_all', 
                   'TMI_H10_all', 'TMI_V10_all', 'TMI_H19_all', 'TMI_V19_all', 'TMI_V21_all', 'TMI_H37_all', 'TMI_V37_all', 'TMI_H85_all', 'TMI_V85_all']        
        self.assign_nan(self.dataset[platform], varlist)
        
        ## store TMI rainType dataset
        Timp = {}
        for var in ['Year', 'Month', 'DayOfMonth', 'Hour', 'LocalHour']:
            Timp[var.lower()] = np.tile(self.dataset[platform][var],(self.dataset[platform]['Latitude'].shape[1], 1)).T.astype(np.int32)
        self.dataset[platform].update(Timp)
        
        varlist = ['Longitude', 'Latitude', 'year', 'month', 'dayofmonth', 'hour', 'localhour', 
                   'rainType', 'nearSurfRain', 'surfRainPhase', 
                   'LandOcean', 'TMI_SurfPrecip', 
                   'TMI_pct10', 'TMI_pct19', 'TMI_pct37', 'TMI_pct85', 'TMI_H10', 'TMI_V10', 'TMI_H19', 'TMI_V19', 'TMI_V21', 'TMI_H37', 'TMI_V37', 'TMI_H85', 'TMI_V85',
                   'VIRS_ch1', 'VIRS_ch2', 'VIRS_ch3', 'VIRS_ch4', 'VIRS_ch5']
        idxs = np.where(self.dataset[platform]['rainType']>0)
        if idxs[0].size > 0:
            fn = '_'.join([platform, 'TMI', 'rain_on_KU_swath', self.dataset[platform]['date']])  # , self.dataset[platform]['orbitID']
            self.data_storage(fn, platform, varlist=varlist, subset=idxs, subdir=self.data_subdirs[-2])
        '''
        ########## VIRS part
        varlist = ['Longitude', 'Latitude', 'year', 'month', 'dayofmonth', 'hour', 'localhour', 'rainType', 'nearSurfRain', 'LandOcean', 
                   'VIRS_ch1', 'VIRS_ch2', 'VIRS_ch3', 'VIRS_ch4', 'VIRS_ch5']
#        idxs = np.where(self.dataset[platform]['rainType']>0)
        if idxs[0].size > 0:
            fn = '_'.join([platform, 'VIRS', 'rainType_on_KU_swath', self.dataset[platform]['date']]) # , self.dataset[platform]['orbitID']
            self.data_storage(fn, platform, varlist=varlist, subset=idxs, subdir=self.data_subdirs[4])
        ###################
        '''
        Timp = {}
        for var in ['Year_tmi', 'Month_tmi', 'DayOfMonth_tmi', 'Hour_tmi', 'LocalHour_tmi']:
            Timp[var.lower().split('_')[0]] = np.tile(self.dataset[platform][var],(self.dataset[platform]['TMI_Lon'].shape[1], 1)).T.astype(np.int32)
        self.dataset[platform].update(Timp)
            
        varlist = ['TMI_Lon', 'TMI_Lat', 'year', 'month', 'dayofmonth', 'hour', 'localhour', 'TMI_SurfPrecip_all', 'LandOcean_all', 
                   'TMI_pct10_all', 'TMI_pct19_all', 'TMI_pct37_all', 'TMI_pct85_all', 
                   'TMI_H10_all', 'TMI_V10_all', 'TMI_H19_all', 'TMI_V19_all', 'TMI_V21_all', 'TMI_H37_all', 'TMI_V37_all', 'TMI_H85_all', 'TMI_V85_all']
        idxs = np.where(self.dataset[platform]['TMI_SurfPrecip_all']>0)
        if idxs[0].size > 0:
            fn = '_'.join([platform, 'TMI', 'rain_on_TMI_swath', self.dataset[platform]['date']]) # , self.dataset[platform]['orbitID']
            self.data_storage(fn, platform, varlist=varlist, subset=idxs, subdir=self.data_subdirs[-1])
        
        [self.dataset[platform].pop(var) for var in ['year', 'month', 'dayofmonth', 'hour', 'localhour', 'Year_tmi', 'Month_tmi', 'DayOfMonth_tmi', 'Hour_tmi', 'LocalHour_tmi',
                      'TMI_H10', 'TMI_V10', 'TMI_H19', 'TMI_V19', 'TMI_V21', 'TMI_H37', 'TMI_V37', 'TMI_H85', 'TMI_V85',
                      'TMI_Lat', 'TMI_Lon', 'LandOcean_all', 'TMI_SurfPrecip_all', 'TMI_pct10_all', 'TMI_pct19_all', 'TMI_pct37_all', 'TMI_pct85_all', 'TMI_H10_all', 'TMI_V10_all', 
                      'TMI_H19_all', 'TMI_V19_all', 'TMI_V21_all', 'TMI_H37_all', 'TMI_V37_all', 'TMI_H85_all', 'TMI_V85_all',
                      'VIRS_ch1', 'VIRS_ch2', 'VIRS_ch3', 'VIRS_ch5']]
        return 
    
    
    def TMI_QA(self, platform):
        idxs = np.where(self.dataset[platform]['TMI__VALIDITY'] > 0)[0]
        if idxs.__len__() > 0:
            keys = set(self.dataset[platform].keys()) - set(['TMI_Lat', 'TMI_Lon', 'LandOcean_all', 'TMI__VALIDITY'])
            for var in keys:
                if 'TMI_' in var:  
                    self.dataset[platform][var][idxs] = np.nan
        
        self.dataset[platform].pop('TMI__VALIDITY')        
        return 
    
    
    def TRMM_1Z_AssignTo_pr_pixels(self, platform):
        dadict = self.dataset[platform]
        keys = set(dadict.keys())
        for var in keys:
            if '_all' in var: dadict[var.split('_all')[0]] = dadict[var].ravel()[dadict['colohi']]
        dadict.pop('colohi') 
        return 
    
    
    def TRMM_1Z_TMI_changeSize(self, platform):
        dadict = self.dataset[platform]
        shpY = dadict['TMI_pct37'].shape[1]
        for var in ['TMI_pct10', 'TMI_pct19', 'TMI_pct37', 'TMI_H10', 'TMI_V10', 'TMI_H19', 'TMI_V19', 'TMI_V21', 'TMI_H37', 'TMI_V37']:
            tmpdata = np.full_like(dadict['TMI_pct85'], -9990)
            tmpdata[:, 2*np.arange(shpY)] = dadict[var]
            tmpdata[:, 2*np.arange(shpY)+1] = dadict.pop(var)
            dadict[var+'_all'] = tmpdata
            
        for var in ['TMI_pct85', 'TMI_H85', 'TMI_V85', 'LandOcean', 'TMI_SurfPrecip']:
            dadict[var+'_all'] = dadict.pop(var)
        return 
    
    
    def assign_nan(self, daset, varlist, thh=0, has_thh=False):
        if has_thh:
            for key in varlist:
                if not isinstance(daset[key], np.float32): daset[key] = daset[key].astype(np.float32)        
                daset[key][daset[key]<=thh] = np.nan
        else:
            for key in varlist:
                if not isinstance(daset[key], np.float32): daset[key] = daset[key].astype(np.float32)    
                daset[key][daset[key]<thh] = np.nan             
        return 
    
    
    def check_var(self, data):
        da, db, dc = '', '', ''
        
        tmp = data.ravel()
        if tmp.size > 0:
            da = stats.mode(tmp)[0][0], tmp.min(), np.median(tmp), tmp.max()  # 众数，最小值，中值，最大值
            
        tmp = data[data>0].ravel()
        if tmp.size > 0:
            db = stats.mode(tmp)[0][0], tmp.min(), np.median(tmp), tmp.max()
            
        tmp = data[data<0].ravel()
        if tmp.size > 0:
            dc = np.unique(tmp)
        
        return da, db, dc

    
   