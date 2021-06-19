# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:28:13 2021

@author: aa
"""
import module_Parameters as param

import numpy as np
import pandas as pd
import vaex as ve

import os
import glob

import h5py
import time


class Analysis_Ku():
    def __init__(self, platform='TRMM', daset=['Rainfall_pixels'], fn_Medata='', times={'month': [], 'year': []}):
        self.platform = platform
        self.daset = daset
        self.fn_Medata = fn_Medata
        self.timedf = self.check_times(times)
        self.grid_gap = f'grid{param.gap[self.platform]:.02f}'
        
        self.statVar = {'Year': 'Year', 'Month': 'Month', 'LocalHour': 'LocalHour', 
                        'MJO_Phase': 'MJO_BSISO_Phase', 'BSISO_Phase': 'MJO_BSISO_Phase'}
        
        self.data_output = os.path.join(param.res_Disk, param.op_res, param.op_data, param.DIR_final, self.fn_Medata.split('pixels_')[1])        
        self.__check_DIR__(self.data_output)
        
        ## switch; 0: ignored; 1: just do it
        self.DO_ALL = 0   # do all calculation
        
        self.all_stat = 0
        self.year_stat = 0
        self.month_stat = 0
        self.diurnal_stat = 0
        self.MJO_stat = 1
        self.BSISO_stat = 1
        
        self.lonlat = 1  # Lat/Lon (2D)
        self.count_Ku = 0   # Number of PR sample
        
        # (1)  dims: lon-lat      
        self.count_conv = 1  # conv. sample
        self.count_rain = 1  # raining sample
        self.count_20dBZ_Htop = 1  # 20dBZ sample
        self.count_20dBZ_conv_Htop = 1  # conv. 20dBZ sample
        self.count_35dBZ_Htop = 1  # 35dBZ sample 
        self.count_35dBZ_conv_Htop = 1  # conv. 35dBZ sample
        
        # (2)  dims: lon-lat  
        self.sum_total_rain = 1  # Total rain
        self.sum_conv_rain = 1   # Total conv. rain
        
        # (3)  dims: lon-lat  
        self.sum_20dBZ_Htop = 1  # sum of 20dBZ Htop heights
        self.sum_20dBZ_conv_Htop = 1  # sum of conv. 20dBZ Htop heights
        self.sum_35dBZ_Htop = 1  # sum of 35dBZ Htop heights
        self.sum_35dBZ_conv_Htop = 1  # sum of conv. 35dBZ Htop heights
        
        # (4)  dims: lon-lat  
        self.percentile = [10, 25, 50, 75, 90, 95]  # xx%
        self.pect_20dBZ_Htop = 1
        self.pect_20dBZ_conv_Htop = 1
        self.pect_35dBZ_Htop = 1
        self.pect_35dBZ_conv_Htop = 1
        
        # (5)  dims: lon-lat  
        self.hgt_20dBZ_LE = [2, 3, 4]   # km
        self.hgt_20dBZ_GE = [6, 8, 10, 12, 14]  # km
        self.count_lev_20dBZ_Htop = 1  # Num of 20dBZ Htop
        self.count_lev_20dBZ_conv_Htop = 1  # Num of conv. 20dBZ Htop
        
        # (6)  dims: lon-lat  
#        self.hgt_35dBZ_LE = [2, 3, 4]   # km
        self.hgt_35dBZ_GE = [5, 6, 7, 8, 9, 10, 11, 12]  # km
        self.count_lev_35dBZ_Htop = 1    # Num of 35dBZ Htop
        self.count_lev_35dBZ_conv_Htop = 1  # Num of conv. 35dBZ Htop
        
        # (7)  dims: lat-height
        self.count_20dBZ_Htop_LatH = 1  # Num of 20dBZ in Lat vs. Height
        self.count_20dBZ_conv_Htop_LatH = 1  # Num of conv. 20dBZ in Lat vs. Height
        
        self.check_switch()
        #####
        return
    
    
    def check_switch(self, ):
        if self.DO_ALL == 1:
            self.all_stat = 1
            self.year_stat = 1
            self.month_stat = 1
            self.diurnal_stat = 1
            self.MJO_stat = 1
            self.BSISO_stat = 1
            
            self.lonlat = 1
            self.count_Ku = 1
            
            # (1)  dims: lon-lat      
            self.count_conv = 1
            self.count_20dBZ_Htop = 1
            self.count_20dBZ_conv_Htop = 1
            self.count_35dBZ_Htop = 1
            self.count_35dBZ_conv_Htop = 1
            
            # (2)  dims: lon-lat  
            self.sum_total_rain = 1
            self.sum_conv_rain = 1
            
            # (3)  dims: lon-lat  
            self.sum_20dBZ_Htop = 1
            self.sum_20dBZ_conv_Htop = 1
            self.sum_35dBZ_Htop = 1
            self.sum_35dBZ_conv_Htop = 1
            
            # (4)  dims: lon-lat  
            self.pect_20dBZ_Htop = 1
            self.pect_20dBZ_conv_Htop = 1
            self.pect_35dBZ_Htop = 1
            self.pect_35dBZ_conv_Htop = 1
            
            # (5)  dims: lon-lat  
            self.count_lev_20dBZ_Htop = 1
            self.count_lev_20dBZ_conv_Htop = 1
            
            # (6)  dims: lon-lat  
            self.count_lev_35dBZ_Htop = 1
            self.count_lev_35dBZ_conv_Htop = 1
            
            # (7)  dims: lat-height
            self.count_20dBZ_Htop_LatH = 1
            self.count_20dBZ_conv_Htop_LatH = 1 
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
            
            if param.DIR_Medata == daset:
                dadict[daset] = pd.Series(glob.glob(os.path.join(DIR, daset, self.fn_Medata+'.h5')))
            else:  
                dadict[daset] = pd.Series(glob.glob(os.path.join(DIR, daset, '*.h5')))
                dadict[daset] = pd.concat([dadict[daset].str.split(r'\.|_', expand=True)[7], dadict[daset]], axis=1)
                dadict[daset].columns = ['date', 'fn']
                dadict[daset] = dadict[daset].merge(self.timedf, left_on='date', right_on='date', how='inner')            
        return dadict
    
    
    def Analyses_handler(self,):
        fndict = self.check_datalist()
        
        self.grid_lls = {}
        self.grid_lls['Lon'], self.grid_lls['Lat'] = np.meshgrid(np.arange(param.limits[self.platform][0][0], param.limits[self.platform][0][-1], param.gap[self.platform])[:-1]+0.5*param.gap[self.platform], 
                              np.arange(param.limits[self.platform][1][0], param.limits[self.platform][1][-1], param.gap[self.platform])[:-1]+0.5*param.gap[self.platform])
        self.grid_lls['Lon'], self.grid_lls['Lat'] = self.grid_lls['Lon'].T, self.grid_lls['Lat'].T
        
        varlist = []
#        ['Latitude', 'Longitude', 'Year', 'Month', 'LocalHour', 'MJO_BSISO_Phase', 'MJO_BSISO_Magnitude']   
        if self.count_Ku == 1:  # Number of PR sample
            if 'ALL_pixels' in fndict.keys():
                self.calc_sample_base(fndict['ALL_pixels'], flag='UnCdn_Sbase', varlist=varlist)     
        
#        if 'Rainfall_pixels' in fndict.keys():
#            Condition_smp_base = self.calc_sample_base(fndict['Rainfall_pixels'], flag='Cdn_Sbase', varlist=varlist)
        
        if param.DIR_Medata in self.daset:
            self.Conditional_pro(fndict[param.DIR_Medata].loc[0], varlist=varlist)
        
        # write self.grid_lls into all h5 files
        if self.lonlat == 1:  # Lat/Lon (2D)
            op_files = glob.glob(os.path.join(self.data_output, self.platform+'*.h5'))
            for h5fn in op_files:
                self.data_storage(self.grid_lls, fn=h5fn, mode='a')        
        return
    
    
    def Conditional_pro(self, fn, varlist=[]):
        flag, res = 'Cdn', {}
        dataset = self.open_daset(fn, varlist)
        
#        ['Longitude', 'Latitude', 'Year', 'Month', 'LocalHour', 'MJO_BSISO_Phase', 'MJO_BSISO_Magnitude', 
#         'rainType', 'LandOcean', 'nearSurfRain', 'nearSurfZ', 'ku_dBZ20_Htop', 'ku_dBZ35_Htop']

        if self.year_stat == 1:
            statKey = 'Year'            
            res.update(self.LonLat_stat(dataset, self.check_valid_OpFiles(statKey), statKey, flag=flag))            
        
        if self.all_stat == 1:
            VarDF = pd.Series(list(res.keys()))
            VarDF = pd.concat([VarDF.str.split('Year_', expand=True)[1], VarDF], axis=1)
            VarDF.columns = ['tail', 'var']
            
            res_all = {}            
            grps = VarDF.groupby('tail')
            for grp, idxs in grps.groups.items():
                if 'pect' not in grp:
                    varname = self.name_system(flag, 'All', grp)
                    res_all[varname] = np.dstack([res[VarDF.loc[idx, 'var']] for idx in idxs]).sum(axis=2)
                    
            # percentage part
            if self.pect_20dBZ_Htop == 1:
                statVar = 'ku_dBZ20_Htop'
                sele = f'({statVar}>0)'
                res_all.update(self.stat_pect(flag, dataset, statVar, sele, 'All', varname_sfx=statKey+'_km_'+statVar))
                        
            if self.pect_35dBZ_Htop == 1:
                statVar = 'ku_dBZ35_Htop'
                sele = f'({statVar}>0)'
                res_all.update(self.stat_pect(flag, dataset, statVar, sele, 'All', varname_sfx=statKey+'_km_'+statVar))
                            
            fn = glob.glob(os.path.join(self.data_output, '_'.join([self.platform, 'All'])+'*.h5'))[0].split('.h5')[0]
            self.data_storage(res_all, fn=fn, mode='a')
            
            del res, res_all
            
        if self.month_stat == 1:
            statKey = 'Month'
            self.LonLat_stat(dataset, self.check_valid_OpFiles(statKey), statKey, flag=flag)            
            
        if self.diurnal_stat == 1:
            statKey = 'LocalHour'
            self.LonLat_stat(dataset, self.check_valid_OpFiles(statKey), statKey, flag=flag)            
            
        if self.MJO_stat == 1:
            statKey = 'MJO_Phase'
            self.LonLat_stat(dataset, self.check_valid_OpFiles(statKey), statKey, flag=flag)            
            
        if self.BSISO_stat == 1:
            statKey = 'BSISO_Phase'
            self.LonLat_stat(dataset, self.check_valid_OpFiles(statKey), statKey, flag=flag)            
        return
    
    
    def check_valid_OpFiles(self, statKey):
        op_files = pd.Series(glob.glob(os.path.join(self.data_output, '_'.join([self.platform, statKey])+'*.h5')))
        op_files = pd.concat([op_files.str.split(f'{statKey}_|_grid', expand=True)[1].str.split('_', expand=True)[0].astype(int).astype(str), op_files], axis=1)
        op_files.columns = [statKey, 'fn']
        return op_files
    
    
    def stat_demo(self, dataset, statvar, statMethod, dtype, sele):
        res = self.generate_2Darray(dataset, statvar, statMethod, dtype=dtype, 
                                    bins=['Longitude', 'Latitude'],
                                    lims=param.limits[self.platform], 
                                    shape=param.shapes[self.platform],
                                    selection=sele)        
        return res
    
    
    def stat_demo2(self, dataset, statvar, statMethod, dtype, sele):
        res = self.generate_2Darray(dataset, statvar, statMethod, dtype=dtype, 
                                    bins=['Latitude', 'ku_dBZ20_Htop'],
                                    lims=param.limits[self.platform+'_Lat'], 
                                    shape=param.shapes[self.platform+'_Lat'],
                                    selection=sele)        
        return res
    
    
    def stat_pect(self, flag, dataset, statVar, sele, key, varname_sfx):
        res, statMethod = {}, 'percentile'
        for pect in self.percentile:
            varname = self.name_system(flag, key, sfx=varname_sfx+f'_pect{pect}')
            
#            Nslice = 2
            Nslice, lim_slice, shp_slice = self.calc_Nslice(dataset.shape[0], param.limits[self.platform], param.shapes[self.platform])
            res[varname] = self.calc_slice_combination(Nslice, dataset, statVar, statMethod, 
                                                       ['Longitude', 'Latitude'], lim_slice, shp_slice, pect, 
                                                       selection=sele)            
        return res
    
    
    def calc_slice_combination(self, Nslice, dataset, statvar, statmethod, bins=['Longitude', 'Latitude'], lim_slice=[], shp=[], 
                               percent=50, selection=False, dtype='float32'):
        res = np.concatenate([self.generate_2Darray(dataset, statvar, statmethod, bins, lim_slice[islice], shp,                                        
                                       percentage=percent, selection=selection, dtype=dtype) 
                              for islice in np.arange(Nslice)], axis=0)        
        return res
    
    
    def calc_Nslice(self, Nsmp, lims, shp):
        xlen = lims[0][1]-lims[0][0]
        if Nsmp > param.stat_pect_Nsample:        
            for Nslice in np.arange(int(np.ceil(Nsmp/param.stat_pect_Nsample)), int(xlen/2)+1, 1):
                if xlen % Nslice < 1.e-5: break
            
            intl_xlim = int(xlen/Nslice)
            lim_slice = [[[lims[0][0]+idx*intl_xlim, lims[0][0]+(idx+1)*intl_xlim], lims[1]] for idx in np.arange(Nslice)]
            lim_slice[-1][0][-1] += 1.e-6
            
            shp_slice = (int(shp[0]/Nslice), shp[1])
        else:
            Nslice, lim_slice, shp_slice = 1, lims, shp
        return Nslice, lim_slice, shp_slice
    
    
    def LonLat_stat(self, dataset, fndf, statKey, flag='', dtype='int32'):  
        grid_all = {}          
        for idx in fndf.index:
            grid = {}
            key, fn = fndf.loc[idx, statKey], fndf.loc[idx, 'fn'].split('.h5')[0]
            
            if statKey in ['LocalHour']:
                Tsele = f'(({self.statVar[statKey]} >= {key}) & ({self.statVar[statKey]} < {str(int(key)+3)}))'
            else:
                Tsele = f'({self.statVar[statKey]} == {key})'
            conv_sele = f'(rainType == {param.raintypes["convective"]})'
            
            
            # (1)  dims: lon-lat
            statVar, statMethod = self.statVar[statKey], 'count'
            
            sele = False
            if self.count_conv == 1:
                sele = ' & '.join([conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_Conv')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
                
            sele = False    
            if self.count_rain == 1:
                sele = ' & '.join(['(nearSurfRain > 0)', Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_Rain')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
                
            sele = False    
            if self.count_20dBZ_Htop == 1:
                sele = ' & '.join(['(ku_dBZ20_Htop > 0)', Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ20_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
                
            sele = False    
            if self.count_20dBZ_conv_Htop == 1:
                sele = ' & '.join(['(ku_dBZ20_Htop > 0)', conv_sele, Tsele]) 
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ20_Conv_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
                
            sele = False    
            if self.count_35dBZ_Htop == 1:
                sele = ' & '.join(['(ku_dBZ35_Htop > 0)', Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ35_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
                
            sele = False    
            if self.count_35dBZ_conv_Htop == 1:
                sele = ' & '.join(['(ku_dBZ35_Htop > 0)', conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ35_Conv_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, dtype, sele)
        
            
            # (2)  dims: lon-lat  
            statVar, statMethod = 'nearSurfRain', 'sum'
            
            sele = False 
            if self.sum_total_rain == 1:
                sele = ' & '.join([f'({statVar} > 0)', Tsele]) 
                varname = self.name_system(flag, key, sfx=statKey+'_mmh_Total_rain')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
                
            sele = False 
            if self.sum_conv_rain == 1:
                sele = ' & '.join([f'({statVar} > 0)', conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_mmh_Total_Conv_rain')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
        
        
            # (3)  dims: lon-lat  
            statVar, statMethod = 'ku_dBZ20_Htop', 'sum'
            
            sele = False 
            if self.sum_20dBZ_Htop == 1:
                sele = ' & '.join([f'({statVar} > 0)', Tsele]) 
                varname = self.name_system(flag, key, sfx=statKey+'_km_ku_dBZ20_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
                
            sele = False 
            if self.sum_20dBZ_conv_Htop == 1:
                sele = ' & '.join([f'({statVar} > 0)', conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_km_ku_dBZ20_Conv_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
                
            sele = False
            statVar = 'ku_dBZ35_Htop'
            if self.sum_35dBZ_Htop == 1:
                sele = ' & '.join([f'({statVar} > 0)', Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_km_ku_dBZ35_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
                
            sele = False 
            if self.sum_35dBZ_conv_Htop == 1:
                sele = ' & '.join([f'({statVar} > 0)', conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_km_ku_dBZ35_Conv_Htop')
                grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'float32', sele)
                
            
            # (5)  dims: lon-lat
            statVar, statMethod = 'ku_dBZ20_Htop', 'count'
            
            sele = False 
            if self.count_lev_20dBZ_Htop == 1:
                for hgt in self.hgt_20dBZ_LE:
                    sele = ' & '.join([f'({statVar}<={hgt})', Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ20_Htop_LE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
                    
                for hgt in self.hgt_20dBZ_GE:
                    sele = ' & '.join([f'({statVar}>={hgt})', Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ20_Htop_GE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
                
            sele = False
            if self.count_lev_20dBZ_conv_Htop == 1:
                for hgt in self.hgt_20dBZ_LE:
                    sele = ' & '.join([f'({statVar}<={hgt})', conv_sele, Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ20_Conv_Htop_LE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
                    
                for hgt in self.hgt_20dBZ_GE:
                    sele = ' & '.join([f'({statVar}>={hgt})', conv_sele, Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ20_Conv_Htop_GE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
            
            
            # (6)  dims: lon-lat  
            statVar, statMethod = 'ku_dBZ35_Htop', 'count'
            
            sele = False 
            if self.count_lev_35dBZ_Htop == 1:
                for hgt in self.hgt_35dBZ_GE:
                    sele = ' & '.join([f'({statVar}>={hgt})', Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ35_Htop_GE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
                
            sele = False
            if self.count_lev_35dBZ_conv_Htop == 1:
                for hgt in self.hgt_35dBZ_GE:
                    sele = ' & '.join([f'({statVar}>={hgt})', conv_sele, Tsele])
                    varname = self.name_system(flag, key, sfx=statKey+f'_Num_ku_dBZ35_Conv_Htop_GE{hgt}km')
                    grid[varname] = self.stat_demo(dataset, statVar, statMethod, 'int32', sele)
            
            # (7)  dims: lat-height
            statVar, statMethod = 'ku_dBZ20_Htop', 'count'
            
            sele = False 
            if self.count_20dBZ_Htop_LatH == 1:
                sele = ' & '.join([f'({statVar}>0)', Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ20_Htop_LatHgt')
                grid[varname] = self.stat_demo2(dataset, statVar, statMethod, 'int32', sele)
            
            sele = False 
            if self.count_20dBZ_conv_Htop_LatH == 1:
                sele = ' & '.join([f'({statVar}>0)', conv_sele, Tsele])
                varname = self.name_system(flag, key, sfx=statKey+'_Num_ku_dBZ20_Conv_Htop_LatHgt')
                grid[varname] = self.stat_demo2(dataset, statVar, statMethod, 'int32', sele)
            
            # (4)  dims: lon-lat
            statVar = 'ku_dBZ20_Htop'
                        
            sele = False 
            if self.pect_20dBZ_Htop == 1:
                sele = ' & '.join([f'({statVar} > 0)', Tsele])                
                grid.update(self.stat_pect(flag, dataset, statVar, sele, key, varname_sfx=statKey+'_km_ku_dBZ20_Htop'))
                
            '''
            sele = False 
            if self.pect_20dBZ_conv_Htop == 1:
                sele = ' & '.join([f'({statVar}>0)', conv_sele, Tsele])
                grid.update(self.stat_pect(flag, dataset, statVar, sele, key, varname_sfx=statKey+'_km_ku_dBZ20_Conv_Htop'))
            '''
            
            statVar = 'ku_dBZ35_Htop'
            sele = False 
            if self.pect_35dBZ_Htop == 1:
                sele = ' & '.join([f'({statVar}>0)', Tsele])
                grid.update(self.stat_pect(flag, dataset, statVar, sele, key, varname_sfx=statKey+'_km_ku_dBZ35_Htop'))
            
            '''
            sele = False 
            if self.pect_35dBZ_conv_Htop == 1:
                sele = ' & '.join([f'({statVar}>0)', conv_sele, Tsele])
                grid.update(self.stat_pect(flag, dataset, statVar, sele, key, varname_sfx=statKey+'_km_ku_dBZ35_Conv_Htop'))
            '''
            
            ## storage        
            self.data_storage(grid, fn=fn, mode='a')
            if statKey in ['Year']: grid_all.update(grid)
        return grid_all
        
    ########################################
    def name_system(self, flag, key, sfx='Count'):
        varname = '_'.join([flag, key, sfx])
        
        return varname
    
    
    def open_daset(self, fn, varlist=[]):
        if varlist.__len__() == 0:
            return ve.open(fn)
        else:
            return ve.open(fn)[varlist]
    
    
    def calc_sample_base(self, fndf, flag='UnCdn', varlist=[]):
        res = {}
                
#        aa=time.time()
        # Year loop
        if self.year_stat == 1:
            statKey = 'Year'
            grp_year = fndf.groupby('year')
            varY = []
            for Y4, fidx in grp_year.groups.items():
                varname = self.name_system(flag, Y4, 'Year_Num')
                varY.append(varname)
                for idx in fidx:
                    dataset = self.open_daset(fndf.loc[idx, 'fn'], varlist)
                                    
                    if varname not in res.keys():
                        res[varname] = self.generate_2Darray(dataset, self.statVar[statKey], 'count', bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform], dtype='int32')
                    else:
                        res[varname] += self.generate_2Darray(dataset, self.statVar[statKey], 'count', bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform], dtype='int32')
                    dataset.close()
                
                fn = '_'.join([self.platform, statKey, Y4, 'grid'+f'{param.gap[self.platform]:.02f}'])
                self.data_storage(res, var=varname, fn=fn, mode='w')
                
            del statKey
#            bb=time.time()
#            print(f'{flag}, Year part costs : {(bb-aa)/60} mins')
        
        ## for ALL if year has been counted
        if self.all_stat == 1:
            varname = self.name_system(flag, 'All', 'Num')
            if varname not in res.keys():
                res[varname] = np.dstack([res[var] for var in varY]).sum(axis=2)
                
                fn = '_'.join([self.platform, 'All', self.grid_gap])
                self.data_storage(res, var=varname, fn=fn, mode='w')
        
        # Month loop
        if self.month_stat == 1:
            statKey = 'Month'
            grp_Month = fndf.groupby('month')
            varM = []
            for M2, fidx in grp_Month.groups.items():
                varname = self.name_system(flag, M2, 'Month_Num')
                varM.append(varname)
                for idx in fidx:
                    dataset = self.open_daset(fndf.loc[idx, 'fn'], varlist)
                                    
                    if varname not in res.keys():
                        res[varname] = self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform])
                    else:
                        res[varname] += self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                              bins=['Longitude', 'Latitude'],
                                                              lims=param.limits[self.platform], 
                                                              shape=param.shapes[self.platform])
                    dataset.close()
                    
                fn = '_'.join([self.platform, statKey, M2, self.grid_gap])
                self.data_storage(res, var=varname, fn=fn, mode='w')
            
            del statKey
#            cc=time.time()
#            print(f'{flag}, Month part costs : {(cc-bb)/60} mins')
        
        ### Diurnal 
        if self.diurnal_stat == 1:
            statKey = 'LocalHour'
            varnames = {hh: self.name_system(flag, f'{hh*3:02.0f}_{(hh+1)*3:02.0f}', 'LocalHour_Num') for hh in np.arange(8)}
            for fn in fndf['fn']:
                dataset = self.open_daset(fn, varlist)
                for hh, varname in varnames.items():
                    sele = f'(LocalHour>={hh*3}) & (LocalHour<{(hh+1)*3})'        
                    if varname not in res.keys():
                        res[varname] = self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                    else:
                        res[varname] += self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                dataset.close()
                
            for hh, varname in varnames.items():
                fn = '_'.join([self.platform, statKey, f'{hh*3}_{(hh+1)*3}', self.grid_gap])
                self.data_storage(res, var=varname, fn=fn, mode='w')
            
            del statKey
#            dd=time.time()
#            print(f'{flag}, Diurnal part costs : {(dd-cc)/60} mins')
        
        ### MJO
        if self.MJO_stat == 1:
            statKey = 'MJO_Phase'
            varnames = {Ph: self.name_system(flag, f'{Ph:01.0f}', 'MJO_Phase_Num') for Ph in np.arange(1,9,1)}
            for fn in fndf['fn']:
                dataset = self.open_daset(fn, varlist)
                for Ph, varname in varnames.items():
                    sele = f'(MJO_BSISO_Phase=={Ph}) & (MJO_BSISO_Magnitude>1) & ((Month>=10) | (Month<=4))'
                    if varname not in res.keys():
                        res[varname] = self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                    else:
                        res[varname] += self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                dataset.close()
                
            for Ph, varname in varnames.items():
                fn = '_'.join([self.platform, statKey, f'{Ph:.0f}', self.grid_gap])
                self.data_storage(res, var=varname, fn=fn, mode='w')
            
            del statKey
#            ee=time.time()
#            print(f'{flag}, MJO part costs : {(ee-dd)/60} mins')
            
        ### BSISO
        if self.BSISO_stat == 1:
            statKey = 'BSISO_Phase'
            varnames = {Ph: self.name_system(flag, f'{Ph:01.0f}', 'BSISO_Phase_Num') for Ph in np.arange(1,9,1)}
            for fn in fndf['fn']:
                dataset = self.open_daset(fn, varlist)
                for Ph, varname in varnames.items():
                    sele = f'(MJO_BSISO_Phase=={Ph}) & (MJO_BSISO_Magnitude>1) & (Month>=5) & (Month<=9)'
                    if varname not in res.keys():
                        res[varname] = self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                    else:
                        res[varname] += self.generate_2Darray(dataset, self.statVar[statKey], 'count', dtype='int32', 
                                                             bins=['Longitude', 'Latitude'],
                                                             lims=param.limits[self.platform], 
                                                             shape=param.shapes[self.platform],
                                                             selection=sele)
                dataset.close()
            
            for Ph, varname in varnames.items():
                fn = '_'.join([self.platform, statKey, f'{Ph:.0f}' , self.grid_gap])
                self.data_storage(res, var=varname, fn=fn, mode='w')
            
            del statKey
#            ff=time.time()
#            print(f'{flag},  BSISO part costs : {(ff-ee)/60} mins')
            
        ###########        
#        gg=time.time()
#        print(f'{flag},  ALL parts cost : {(gg-aa)/60} mins')
        return res
    
    
    def generate_2Darray(self, dataset, varname, method, bins=[], lims=[], shape=[], percentage=50, selection=False, dtype='float32'):
        if method in ['sum']:
            data2D = dataset.sum(dataset[varname], binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['count']:
            data2D = dataset.count(dataset[varname], binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['mean']:
            data2D = dataset.mean(dataset[varname], binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['std']:
            data2D = dataset.std(dataset[varname], binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['median']:
            data2D = dataset.median_approx(dataset[varname], binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['percentile']:
            data2D = dataset.percentile_approx(dataset[varname], percentage=percentage, binby=bins, limits=lims, shape=shape, 
                                 selection=selection).astype(dtype)
        elif method in ['div']:
            data2D = np.zeros(shape, dtype=dtype)
            np.true_divide(dataset[varname[0]].astype(dtype), dataset[varname[1]].astype(dtype), 
                                   where=dataset[varname[1]]>0, out=data2D)
        
        if True in np.isnan(data2D):
            data2D[np.isnan(data2D)] = 0
        return data2D
    
    
    def data_storage(self, dadict, var='', fn='test', DIR='', mode='w', aimkey='', simple=False, unit=''):
        if len(DIR) == 0: 
            DIR = self.data_output
        self.__check_DIR__(DIR)
                
        loc = os.path.join(DIR, fn+'.h5')
        store = h5py.File(loc, mode)
        
        if var.__len__() > 0:
            if var not in store.keys():
                store.create_dataset(var, dadict[var].shape, data=dadict[var], compression="gzip", compression_opts=9)
        else:
            for var in dadict.keys():
                if var not in store.keys():
                    store.create_dataset(var, dadict[var].shape, data=dadict[var], compression="gzip", compression_opts=9)
        store.close()
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    