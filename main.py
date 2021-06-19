# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:04:51 2020
https://blog.csdn.net/c2366994582/article/details/80181847      exec
@author: Leo_Ji
"""
from module_Parameters import DIR_Medata

from module_Build_dataset import Satellite_Ku
from module_Analysis_dataset import Analysis_Ku
from module_Concat_dataset import Concat_dataset


def main():
#    cmpKu = Satellite_Ku(mode='new')
#    cmpKu.Manage_handler()
    
    platform = 'GPM'
    
#    cctKu = Concat_dataset(platform=platform, daset=['Rainfall_pixels'], times={'month': [6,7], 'year': [2014, 2016]})  ## times={'month': [4,5]}
#    cctKu.Concat_handler()
    
    anaKu = Analysis_Ku(platform=platform, daset=['ALL_pixels', DIR_Medata], fn_Medata='GPM_Rainfall_pixels_ku_dBZ_Htop', 
                        times={'month': [6,7], 'year': [2014, 2016]})  ## times={'month': [6,7,8]}
    anaKu.Analyses_handler()
    
    print('finish')
    return

    
if __name__ == '__main__':
    main()
    