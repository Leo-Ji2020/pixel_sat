# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:51:29 2021

@author: aa
"""
import module_Parameters as param

import numpy as np
import pandas as pd
import vaex as ve

import os
import glob
import sys

import h5py


class Concat_dataset():
    def __init__(self, platform='TRMM', daset=['Rainfall_pixels'], times={'month': [], 'year': []}):
        self.platform = platform
        self.daset = daset
        self.timedf = self.check_times(times)
        
        self.data_output = os.path.join(param.res_Disk, param.op_res, param.op_data, self.platform, param.DIR_Medata)        
        self.__check_DIR__(self.data_output)
        self.storage_checked = False
        self.data_end_index = 0
        return
    
    
    def check_times(self, times):
        Tdict = {}     
        
        var = 'month'
        if (var not in times.keys()) or (times[var].__len__() == 0) :
            Tdict[var] = pd.DataFrame({var: sorted([f'{ii:02.0f}' for ii in param.Months])})
        else:
            Tdict[var] = pd.DataFrame({var: sorted([f'{ii:02.0f}' for ii in set(times[var]) & param.Months])})
        Tdict[var]['tmp'] = 1
        
        var = 'year'
        if (var not in times.keys()) or (times[var].__len__() == 0):
            Tdict[var] = pd.DataFrame({var: sorted([f'{ii:04.0f}' for ii in param.Years[self.platform]])})
        else:
            Tdict[var] = pd.DataFrame({var: sorted([f'{ii:04.0f}' for ii in set(times[var]) & set(param.Years[self.platform])])})
        Tdict[var]['tmp'] = 1
        
        Tdict = Tdict['year'].merge(Tdict['month'], left_on='tmp', right_on='tmp', how='outer').drop('tmp', axis=1)
        Tdict['date'] = Tdict.apply(lambda v: v['year']+v['month'], axis=1).to_frame() 
        
        return Tdict
    
    
    def __check_DIR__(self, DIR, init=False, check_fn=True):
        if not os.path.exists(DIR): os.makedirs(DIR)
    
    
    def check_datalist(self, ):
        DIR = os.path.join(param.res_Disk, param.op_res, param.op_data, self.platform)
        dadict = {}
        for daset in self.daset:
            dadict[daset] = pd.Series(glob.glob(os.path.join(DIR, daset, '*.h5')))
            dadict[daset] = pd.concat([dadict[daset].str.split(r'\.|_', expand=True)[7], dadict[daset]], axis=1)
            dadict[daset].columns = ['date', 'fn']
            dadict[daset] = dadict[daset].merge(self.timedf, left_on='date', right_on='date', how='inner')            
        return dadict
    
    
    def Concat_handler(self, ):
        dadict = self.check_datalist()
        
        res = {}
        res['Lon'], res['Lat'] = np.meshgrid(np.arange(param.limits[self.platform][0][0], param.limits[self.platform][0][-1], param.gap[self.platform])[:-1]+0.5*param.gap[self.platform], 
                              np.arange(param.limits[self.platform][1][0], param.limits[self.platform][1][-1], param.gap[self.platform])[:-1]+0.5*param.gap[self.platform])
        res['Lon'], res['Lat'] = res['Lon'].T, res['Lat'].T
        
        varlist = ['Longitude', 'Latitude', 'Year', 'Month', 'LocalHour', 'MJO_BSISO_Phase', 'MJO_BSISO_Magnitude', 
                   'rainType', 'LandOcean', 'nearSurfRain', 'nearSurfZ', 'ku_dBZ20_Htop', 'ku_dBZ35_Htop']
        self.Concat_process(dadict, varlist=varlist, base_fn='ku_dBZ_Htop', flag_var='nearSurfRain')
                   
        return
    
    
    def open_daset(self, fn, varlist=[]):
        if varlist.__len__() == 0:
            return ve.open(fn)
        else:
            return ve.open(fn)[varlist]
    
    
    def Concat_process(self, dadict, varlist=[], base_fn='', flag_var=''):
        for datakind in dadict.keys():
            for fn in dadict[datakind]['fn']:
                dataset = self.open_daset(fn, varlist)
                if flag_var.__len__() > 0:
                    self.dataset = dataset[dataset[flag_var]>0]
                else:
                    self.dataset = dataset
                self.data_storage('_'.join([self.platform, datakind, base_fn]), varlist=varlist)
        return
    
    
    def data_storage(self, fn, varlist=(), subdir=''):
        s1 = self.dataset.shape[0] 
        dadict = self.dataset.to_pandas_df()
        fDIR = self.data_output
        sfn = os.path.join(fDIR, fn+'.h5')
    
        ###### check the file open mode and the start point of current dataset
        if not self.storage_checked:
            self.data_end_index = 0
            s0 = 0     #, s0Y = 0
            self.storage_checked = True
            
        else:  # ['append']
            if self.storage_checked:
                s0 = self.data_end_index
            else:
                datmp = ve.open(sfn)
                s0 = datmp.shape[0]
                datmp.close()
                self.storage_checked = True
        
        #### write dataset
        store_pixel = h5py.File(sfn, 'a')                   
        if 'data' in store_pixel.keys(): 
            grp = store_pixel['data']
        else:
            grp = store_pixel.create_group('data')
                
        s1 += s0           
        for key in varlist:  
            if key in grp.keys():
                datas = grp[key]
            else:
                datas = grp.create_dataset(key, [param.HDF5_chunkSize], maxshape=[None], compression="lzf", chunks=True, dtype=np.float32) #, compression_opts=9    
                
            datas.resize(s1, axis=0)
            datas[s0:s1] = dadict[key]
                
        store_pixel.close()            
        self.data_end_index = s1
        return 
    
    
    
    
    
    
    
    
    
    
    
    