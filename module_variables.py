# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:48:25 2021

@author: aa
"""
import module_Parameters as param
from module_Plot import Plotting

import numpy as np
import pandas as pd

from scipy.ndimage import label, generate_binary_structure #, center_of_mass
#from scipy.spatial.distance import pdist, squareform
#from scipy.optimize import curve_fit

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#from sklearn.cluster import DBSCAN as dbscan
from pyclustering.cluster.dbscan import dbscan

#from geopy import distance, Point
from shapely import geometry  #, affinity
from shapely.ops import nearest_points
import geopandas as gpd

import h5py
import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'


##################################################################
class Ellipse_type():
    def __init__(self, lons=np.empty(0), lats=np.empty(0)):
        self.f_decimal = np.float32    # numerical accuracy
        self.points = np.vstack((lons, lats)).astype(self.f_decimal)
        self.min_eps = 1.e-8
        return
    
    
    def fit_ellipse(self):
        if self.points.shape[-1] <= 2:
            center, param_axis, param_orien = self.points.mean(axis=1), [0.,0.], 0.
        else:
            res = self.get_ellipse(self.points)
            center, param_axis, param_orien = self.check_ellipse(res)    
#        return center, param_axis, param_orien
        return center[0], center[1], param_axis[0], param_axis[1], param_orien
    
    
    def get_ellipse(self, data):
        # https://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python
        center = data.mean(axis=1)
#        if True in (center>180): center[center>180] -= 360.
        
#        sigma = np.cov(data, rowvar=0)
        sigma = np.cov(data)
        
        # compute eigenvalues and associated eigenvectors
        vals, vecs = np.linalg.eigh(sigma)
        if vals.min() < param.min_axis:  # min 
            vals[vals < param.min_axis] = param.min_axis
        
        # eigenvalues give length of ellipse along each eigenvector
        major, minor = 4 * np.sqrt(vals)  #### this is the whole lengths of axises, not the half lengths
        
        # compute "tilt" of ellipse using first eigenvector
        x, y = vecs[:, 0]
        orientation = np.arctan2(y, x)
        
        return [center, [major, minor], orientation]
    
    
    def check_ellipse(self, ellipses):
        
#        param_axises, param_oriens = np.array(ellipses[1]).T.astype(self.f_decimal), np.array(ellipses[2]).T.astype(self.f_decimal)
        if ellipses[1][0] < ellipses[1][1]:
            ellipses[1][0], ellipses[1][1] = ellipses[1][1].astype(self.f_decimal), ellipses[1][0].astype(self.f_decimal)   # major axis should be larger than minor axis
            ellipses[2] -= np.sign(ellipses[2])*0.5*np.pi    # at this time, orientation should be turn 0.5*np.pi
                        
        ellipses[2] = self.constraint_orientation(ellipses[2])   # constraint orientation in [-0.5*np.pi, 0.5*np.pi]
        return ellipses  # centers, param_axises, param_oriens
    

    def constraint_orientation(self, oriens):
        if np.abs(oriens) >= 2*np.pi:  oriens %= 2*np.pi    # [-2*np.pi , 2*np.pi ]
        if np.abs(oriens) >= 1.5*np.pi: oriens -= np.sign(oriens)*2*np.pi     # [-np.pi, np.pi]
        if np.abs(oriens) >= 0.5*np.pi: oriens -= np.sign(oriens)*np.pi     # [-np.pi, np.pi]
        
        return np.degrees(oriens).astype(self.f_decimal)
###########################################################################################################
class Group_block_magnet():
    def __init__(self, map2D, locX, locY, features, pixel_area, raintype, rainfall, surfdBZ, Clusters_whole, obs_edge):
        self.Base2Ddata = map2D
        self.Base2Dwhole = Clusters_whole
        self.obs_edge = obs_edge
        self.locX, self.locY = locX, locY
        self.features = features
        self.pixel_area = pixel_area
        self.raintype = raintype
        self.rainfall = rainfall
        self.surfdBZ = surfdBZ
        
        self.adjoin_dataset, self.adjoin_IDs = {}, {}
        
        self.dist_max = 0.2  # [degree]
        self.half_angle_max = 45
        self.area_min = 100   #[km2]
        self.area_shape_style_check = 500   #[km2]
        self.major_shape_style_check = 0.7  #[deg]  0.8
        self.axisR_shape_style_check = 0.3  # 轴比
        self.action_2deg_line_coef = 0.3   # 空间散点拟合曲线的二次项系数，代表曲率，需要修正方位角的临界值
        self.action_axis_ratio = 0.8
        self.action_area_ratio = 6
        
        self.angle_right = np.deg2rad(90)
        self.dist_gap = 0.05  # [degree]
        self.save_blank = param.save_blank
        return 
    
    ##### fit 2-degree FUNCTION
    ##  y = a1*X**2 + a2*X + a3
    '''
    ######  scipy method
    def Fun(self, X,a1,a2,a3):   # 定义拟合函数形式
        return a1*X**2 + a2*X + a3
    
    
    def error(self, p,X, Y):     # 拟合残差
        return self.Fun(p,X)-Y
    
    
    def get_parameters(self, X, Y):
        ## return [a1, a2, a3]
        para, pcov = curve_fit(self.Fun, X, Y)
        return para
    '''
    ###   sklearn method
    def polynomial_fit(self, xp, yp, deg=2, center=(0,0), orient=0):
        ## return [a3, a2, a1]
        polys = gpd.GeoSeries([geometry.Point(center), geometry.MultiPoint(np.array([xp, yp]).T)])
        Xp, Y, trans_pts = self.sample_transform(polys, orient)
        
        X = PolynomialFeatures(deg).fit_transform(Xp.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X, Y)
        return model.coef_, trans_pts
    
    
    def sample_transform(self, polys, orient=0):           
        newpts = polys.rotate(angle=-orient, origin=polys[0], use_radians=True)     
        return newpts[1:].x.values, newpts[1:].y.values, newpts
    
    #####
    def filter_branch_sample0(self, iclu):   ## for sklearn package
        idxs = np.where(self.Base2Ddata==iclu)
        db = dbscan(eps=self.dist_gap*1.2, min_samples=5)
        db.fit(np.array([self.locX[idxs], self.locY[idxs]]).T)              
#        dbscan_plot(db, pts)
        TF_idxs = db.labels_>=0
        return idxs[0][TF_idxs], idxs[1][TF_idxs]
    
    
    def filter_branch_sample(self, iclu):   # for pyclustering package
        idxs = np.where(self.Base2Ddata==iclu)
        db = dbscan(data=np.array([self.locX[idxs], self.locY[idxs]]).T, eps=self.dist_gap*1.2, neighbors=4) 
        db.process()             
#        dbscan_plot(db, pts)
        if len(db.get_clusters()) > 0:
            IDs = np.concatenate(db.get_clusters(), axis=0)
        else:
            return np.empty(0), np.empty(0)
        return idxs[0][IDs], idxs[1][IDs]
    
    
    def group_clusters(self, dist_matrix):   # for pyclustering package
        db = dbscan(data=dist_matrix, eps=2, neighbors=1, data_type='distance_matrix') 
        db.process()            
        return db.get_clusters(), db.get_noise()
    
    
    def expand_features(self, ):
        self.features['axis_ratio'] = self.features['minor'].truediv(self.features['major']).fillna(1)
        self.features['orient'] = self.features['orient'].apply(np.deg2rad)
        
        self.features['half_angle_window'] = np.deg2rad(self.half_angle_max*(self.features['axis_ratio'])**0.333)
        self.features.loc[self.features.query("axis_ratio > @self.action_axis_ratio").index.to_numpy(), 'half_angle_window'] = self.angle_right
        self.features.loc[self.features.query("area < @self.area_min").index.to_numpy(), 'half_angle_window'] = self.angle_right
        
        valid_TF = np.isin(self.Base2Ddata, self.features.index.to_numpy())        
        self.features['max_SurfdBZ'] = pd.DataFrame({'idxs': self.Base2Ddata[valid_TF], 'dBZ': self.surfdBZ[valid_TF]}).groupby('idxs').max()
        
        self.features['extend_len'] = self.features['major']*0.333
        self.features.loc[self.features.query("extend_len < @self.dist_gap").index.to_numpy(), 'extend_len'] = self.dist_gap
        self.features.loc[self.features.query("area <= @self.pixel_area").index.to_numpy(), 'extend_len'] = self.dist_gap*0.6
        self.features.loc[self.features.query("extend_len > @self.dist_max").index.to_numpy(), 'extend_len'] = self.dist_max
        
        thh = 2.5*self.dist_gap
        self.features.loc[self.features.query("(extend_len < @thh) & (max_SurfdBZ >= 40)").index.to_numpy(), 'extend_len'] = thh
        
#        self.features.sort_values(by='area', ascending=False, inplace=True)
#        self.features.reset_index(inplace=True, drop=True)
#        self.clusters = {ii: self.features.loc[ii, 'ID'] for ii in self.features.index}  #####jjjj
        return
    
    
    def calc_distance_matrix(self, polys):  # 找节点
        valid_idxs = self.features.loc[(self.features['area']>=self.area_min)].index.to_numpy()
        self.Mdist = np.zeros((polys.size, polys.size), dtype=np.float32)
        if valid_idxs.__len__() > 1:
            self.Mdist[np.isin(self.ID_arr, valid_idxs)] = np.array([polys.distance(polys[iclu]).values for iclu in valid_idxs], dtype=np.float32)  
            self.Mdist[self.Mdist>self.dist_max] = 0
            
            ## eliminate dup pairs
            TF_tmp = (self.Mdist.T * self.Mdist) > 0
            self.Mdist[np.where(np.tril(TF_tmp) > 0)] = 0
            
            ## constrain all protntial connected-clusters should contain in the SAME rain GROUP
            valid_idxs = np.isin(self.Base2Ddata, self.ID_arr)
            res = (pd.DataFrame({'conv': self.Base2Ddata[valid_idxs], 'whole': self.Base2Dwhole[valid_idxs]})
                               .query('whole>0')
                               .drop_duplicates(keep='first')
                               .assign(temp=0)  # 设置相同值的列用于全连接
                               .pipe(lambda df: df.merge(df, on='temp'))   # 全连接
                               .pipe(lambda df: df.assign(dist=df['whole_x']-df['whole_y']))   # 在一起的conv group 距离 = 0
                               .pivot_table(values='dist', index='conv_x', columns='conv_y')).to_numpy()  # 转 2D矩阵
            res[res!=0] = -1
            self.Mdist *= (res+1)
            del res, TF_tmp, valid_idxs        
        return self.Mdist.max()
    
    
    def check_nearestPoint(self, df, base_polys): # 进一步检查距离，是否出现最近点属于非端点的情况   
        idxs = df.query('Bingo == -2').index
        for idx in idxs:
            pts = nearest_points(base_polys[df.loc[idx,'p0']], base_polys[df.loc[idx,'p1']])
            if df.loc[idx, 'half_angle_window'] < self.angle_right:
                if df.loc[idx, 'half_angle_window'] < self.check_union(pt0=geometry.Point([df.loc[idx, 'centerLon'], df.loc[idx, 'centerLat']]), pt1=pts[1], 
                                                                       pt2=geometry.Point([df.loc[idx, 'centerLon']+ df.loc[idx, 'major']*np.cos(df.loc[idx, 'orient']), 
                                                                                           df.loc[idx, 'centerLat']+ df.loc[idx, 'major']*np.sin(df.loc[idx, 'orient'])])):
                    df.loc[idx, 'Bingo'] = -11
                        
            if df.loc[idx, 'Bingo'] == -2:
                if df.loc[idx, 'half_angle_window1'] >= self.angle_right: continue
                if df.loc[idx, 'half_angle_window1'] < self.check_union(pt0=geometry.Point([df.loc[idx, 'centerLon1'], df.loc[idx, 'centerLat1']]), pt1=pts[0], 
                                                                        pt2=geometry.Point([df.loc[idx, 'centerLon1']+ df.loc[idx, 'major1']*np.cos(df.loc[idx, 'orient1']), 
                                                                                            df.loc[idx, 'centerLat1']+ df.loc[idx, 'major1']*np.sin(df.loc[idx, 'orient1'])])):
                    df.loc[idx, 'Bingo'] = -11
        return 
    
    
    def Filter_cluster_pairs(self, base_polys):   # 剪枝
        idxs = np.where(self.Mdist>0)        
        df = pd.DataFrame({'p0': self.ID_arr[idxs[0]], 'p1': self.ID_arr[idxs[1]], 'dist': self.Mdist[idxs]})
        df['Bingo'] = -1   # record the adjoin status, predefine= All seperated
        
        ## 剪枝：距离
        df = pd.merge(df, self.features[['extend_len']], left_on='p0', right_index=True, how='left')
        df = pd.merge(df, self.features[['extend_len']], left_on='p1', right_index=True, how='left', suffixes=['','1'])   
        df['distX'] = df.apply(lambda v: max(v['extend_len'], v['extend_len1'])-v['dist'], axis=1)
#        df = df.query('distX>=0')[['p0', 'p1']]   # first step finish, BASE VERSION
        df.loc[df.query('distX>=0').index, 'Bingo'] = -2   # first step finish
        
        ## 剪枝：夹角
        cols = ['centerLon', 'centerLat', 'major', 'minor', 'orient', 'area','axis_ratio', 'half_angle_window']
        df = pd.merge(df, self.features[cols], left_on='p0', right_index=True, how='left')
        df = pd.merge(df, self.features[cols], left_on='p1', right_index=True, how='left', suffixes=['','1'])
        df['area_ratio'] = (df['area'] / df['area1'])
        
        # modify half_angle_window
#        tmp = df.query('(area_ratio >= @self.action_area_ratio) & (axis_ratio<1) & (axis_ratio1<1)')[['area', 'area1']]
        tmp = df.query('(Bingo == -2) & (area_ratio >= @self.action_area_ratio) & (axis_ratio<1) & (axis_ratio1<1)')[['area', 'area1', 'area_ratio']]
        if tmp.index.size > 0:
            df.loc[tmp.query('(area_ratio<=0.1) | (area<=150)').index, 'half_angle_window'] = self.angle_right
            df.loc[tmp.query('(area_ratio>=10) | (area1<=150)').index, 'half_angle_window1'] = self.angle_right
        del tmp
        df['area_ratio'] = df['area_ratio'].apply(lambda x: 1/x if x<1 else x)
        
        tmp = df.query('(Bingo == -2) & (dist > 0.12) & ((half_angle_window > (@self.angle_right-0.0001)) | (half_angle_window1 > (@self.angle_right-0.0001))) & ((area<=150) | (area1<=150))')
        if tmp.index.size > 0: df.loc[tmp.index, 'Bingo'] = -10
        
        self.check_nearestPoint(df, base_polys)
        
#        df = self.check_shapestyle(df, base_polys)        ## check if the pf stand for line style rainfall        
        
        # calculate angle between the connection of centers and it major axis         
        df['conn_angle0'] = df.apply(lambda v: self.check_union(pt0=geometry.Point([v['centerLon'], v['centerLat']]), 
                                                                pt1=geometry.Point([v['centerLon1'], v['centerLat1']]),  
                                                                pt2=geometry.Point([v['centerLon']+v['major']*np.cos(v['orient']), 
                                                                                    v['centerLat']+v['major']*np.sin(v['orient'])])) 
                                                                if v['major']>0 else 0, axis=1)
        df['conn_angle1'] = df.apply(lambda v: self.check_union(pt0=geometry.Point([v['centerLon1'], v['centerLat1']]), 
                                                                pt1=geometry.Point([v['centerLon'], v['centerLat']]),  
                                                                pt2=geometry.Point([v['centerLon1']+v['major1']*np.cos(v['orient1']), 
                                                                                    v['centerLat1']+v['major1']*np.sin(v['orient1'])]))
                                                                if v['major1']>0 else 0, axis=1)        
#        df = df.query('(conn_angle0<=half_angle_window) & (conn_angle1<=half_angle_window1)')[['p0', 'p1']]  # second step finish
        df.loc[df.query('(Bingo == -2) & (conn_angle0<=half_angle_window) & (conn_angle1<=half_angle_window1)').index, 'Bingo'] = -3  # second step finish

        ## 执行剪枝操作
#        cut_idxs = np.setdiff1d(np.arange(len(idxs[0])), df.index)
        cut_idxs = df.query('Bingo != -3').index
        self.Mdist[idxs[0][cut_idxs], idxs[1][cut_idxs]] = 0        
        del idxs, cut_idxs
        return df
    
    '''
    def check_shapestyle(self, df, base_polys):
        dftmp = df.query('Bingo == -2')[['p0', 'p1']]
        stf_idx = (set(dftmp['p0']) | set(dftmp['p1'])) & set(self.features.query(
            '(area>=@self.area_shape_style_check) & (major >= @self.major_shape_style_check) & (axis_ratio >= @self.axisR_shape_style_check)').index)
        del dftmp
        
        for iclu in stf_idx:
            valid_idxs = self.filter_branch_sample(iclu)    
            if valid_idxs[0].size > 20:
                coef, trans_pts = self.polynomial_fit(self.locX[valid_idxs], self.locY[valid_idxs], center=self.features.loc[iclu, ['centerLon','centerLat']],
                                                  orient=self.features.loc[iclu, 'orient'])
#                if abs(coef[-1]) >= self.action_2deg_line_coef: 
                df = self.modify_features(df, iclu, trans_pts, base_polys, deg2_coef=coef)
        return df
    
    
    def modify_features(self, df, iclu, pts, base_polys, deg2_coef=[]):
        old_orient = self.features.loc[iclu, 'orient']
        X, Y = pts[1].x.values, pts[1].y.values
        x_rngs = X.min(), X.max()
        pct = min(2 + abs(deg2_coef[-1]), 4)
        xa, xb = x_rngs[0]+(x_rngs[-1]-x_rngs[0])/pct, x_rngs[-1]-(x_rngs[-1]-x_rngs[0])/pct
        idxs_L = np.where(X < xa)
        idxs_R = np.where(X > xb)
        
        new_polys = gpd.GeoSeries([pts[0].values[0], 
                                   geometry.MultiPoint(np.array([X[idxs_L], Y[idxs_L]]).T), 
                                   geometry.MultiPoint(np.array([X[idxs_R], Y[idxs_R]]).T)], 
                                   name='new')
#        del idxs_L, idxs_R
        _,_, Tpts = self.sample_transform(new_polys, -old_orient)
        part_pts = [np.array([Tpts[1].x.values, Tpts[1].y.values]), 
                    np.array([Tpts[2].x.values, Tpts[2].y.values])]
        new_polys = gpd.GeoSeries([geometry.MultiPoint(part_pts[0].T), 
                                   geometry.MultiPoint(part_pts[1].T)], name='new')
        
        centers = new_polys.centroid        
        orients = [np.arctan(np.polyfit(X[idxs_L], Y[idxs_L], deg=1)[0])+old_orient, 
                   np.arctan(np.polyfit(X[idxs_R], Y[idxs_R], deg=1)[0])+old_orient] 
        for idx, orient in enumerate(orients):
            if orient > self.angle_right: 
                orients[idx] = orient - np.pi
            elif orient < -self.angle_right:
                orients[idx] = orient + np.pi
        
        for ii in df.query('(Bingo == -2) & (p0==@iclu)').index:
            ID = new_polys.distance(base_polys[df.loc[ii,'p1']]).values.argmin()                        
            df.loc[ii, ['centerLon', 'centerLat', 'orient']] = centers[ID].x, centers[ID].y, orients[ID]
        
        for ii in df.query('(Bingo == -2) & (p1==@iclu)').index:
            ID = new_polys.distance(base_polys[df.loc[ii,'p0']]).values.argmin()         
            df.loc[ii, ['centerLon1', 'centerLat1', 'orient1']] = centers[ID].x, centers[ID].y, orients[ID]
        
        return df
    '''
    '''
    def modify_features0(self, df, iclu, pts, base_polys):        
        old_orient = self.features.loc[iclu, 'orient']
        idxs_L = np.where(pts[1].x.values < pts[0].x.values[0])
        idxs_R = np.where(pts[1].x.values > pts[0].x.values[0])
        
        new_polys = gpd.GeoSeries([pts[0].values[0], 
                                   geometry.MultiPoint(np.array([pts[1].x.values[idxs_L], pts[1].y.values[idxs_L]]).T), 
                                   geometry.MultiPoint(np.array([pts[1].x.values[idxs_R], pts[1].y.values[idxs_R]]).T)], 
                                   name='new')
#        del idxs_L, idxs_R
        _,_, Tpts = self.sample_transform(new_polys, -old_orient)
        
        part_pts = [np.array([Tpts[1].x.values, Tpts[1].y.values]), 
                    np.array([Tpts[2].x.values, Tpts[2].y.values])]
        new_polys = gpd.GeoSeries([geometry.MultiPoint(part_pts[0].T), 
                                   geometry.MultiPoint(part_pts[1].T)], name='new')
        for ii in df.query('p0==@iclu').index:
            ID = new_polys.distance(base_polys[df.loc[ii,'p1']]).values.argmin()            
#            mod_varname = [var if ID==0 else var+f'{ID:.0f}' for var in ['centerLon', 'centerLat', 'orient']]
            elli_instance = Ellipse_type(lons=part_pts[ID][0], lats=part_pts[ID][1])
            res = elli_instance.fit_ellipse()
            del elli_instance
            
            df.loc[ii, ['centerLon', 'centerLat', 'orient']] = res[0], res[1], np.deg2rad(res[-1])
            
        for ii in df.query('p1==@iclu').index:
            ID = new_polys.distance(base_polys[df.loc[ii,'p0']]).values.argmin()         
            
            elli_instance = Ellipse_type(lons=part_pts[ID][0], lats=part_pts[ID][1])
            res = elli_instance.fit_ellipse()
            del elli_instance
            
            df.loc[ii, ['centerLon1', 'centerLat1', 'orient1']] = res[0], res[1], np.deg2rad(res[-1])
        return df
    '''
    
    def arrange_data(self, ):
        self.expand_features()
        polys = self.group_samples()
        if isinstance(polys, int): 
            return -1, -1, -1, -1 ,{} ,-1
        self.ID_arr = polys.index.to_numpy()
        self.features = self.features.loc[self.ID_arr]
        max_dist = self.calc_distance_matrix(polys)
        if max_dist > 0:
            coupled_pf_properties = self.Filter_cluster_pairs(polys)
#            blank_polys = self.group_blank_polys()
#            self.set_psudo_points(polys, blank_polys)
            cluster_distribution, n_cluster, adjoined_pf_map, coupled_pf_properties = self.final_adjust_clusters(coupled_pf_properties)
        else:
            cluster_distribution, n_cluster, adjoined_pf_map, self.adjoin_dataset, self.adjoin_IDs, coupled_pf_properties = -1, -1, -1, -1 ,{} ,-1        
        return cluster_distribution, n_cluster, adjoined_pf_map, self.adjoin_dataset, self.adjoin_IDs, coupled_pf_properties
    
    
    def final_adjust_clusters(self, df_cpfp):
        dist_matrix = (self.Mdist>0).astype(np.int8)
        dist_matrix[dist_matrix==0] = 10
        grp_clusMap, grp_single = self.group_clusters(dist_matrix)
        
        group_map = self.Base2Ddata.copy()
        adjoined_pf_map = np.zeros_like(group_map)
        if len(grp_clusMap) > 0:
            dftmp = df_cpfp.query('Bingo==-3')[['p0', 'p1']]
            for igrp in grp_clusMap:
                iIDs = self.ID_arr[igrp].tolist()
                if np.where(self.features.loc[iIDs, 'area'] > self.area_min)[0].size > 1:  # Filter, 滤除仅融合较小cluster的“降水簇”
                    TF_map, newID = np.isin(group_map, iIDs), self.features.loc[iIDs, 'area'].idxmax()
                    
#                    if self.features.loc[newID, 'half_angle_window'] >= self.angle_right-param.min_eps*10000: 
                    if self.features.loc[newID, 'axis_ratio'] > 0.4:                        
                        if self.features.loc[newID, 'axis_ratio'] > 0.55:
                            df_cpfp.loc[dftmp.query('(p0==@iIDs) | (p1==@iIDs)').index, 'Bingo'] = -4
                            continue
                        elif self.features.loc[newID, 'area'] <= self.area_shape_style_check:
                            df_cpfp.loc[dftmp.query('(p0==@iIDs) | (p1==@iIDs)').index, 'Bingo'] = -4
                            continue
                        elif self.features.loc[newID, 'area']>1000:
                            res = self.features.loc[set(iIDs)-{newID}, ['area', 'axis_ratio']]
                            res = ((res['area']/self.features.loc[newID, 'area'])>=0.8) & (res['axis_ratio']<=0.3)                            
                            if True not in res:
                                df_cpfp.loc[dftmp.query('(p0==@iIDs) | (p1==@iIDs)').index, 'Bingo'] = -4
                                continue  
                     
                    group_map[TF_map] = newID
                    adjoined_pf_map[TF_map] = newID
                    self.adjoin_dataset[str(newID)] = self.cut_cluster(TF_map, group_map)
                    MJ = RA = np.empty(0)
                    for ID in iIDs:
                        idxs = dftmp.query('(p0==@ID) | (p1==@ID)').index
                        df_cpfp.loc[idxs, 'Bingo'] = 0
                        MJ = np.concatenate((MJ, df_cpfp.loc[idxs, 'major'].values, df_cpfp.loc[idxs, 'major1'].values), axis=0)
                        RA = np.concatenate((RA, df_cpfp.loc[idxs, 'axis_ratio'].values, df_cpfp.loc[idxs, 'axis_ratio1'].values), axis=0)                        
                        
                    self.adjoin_IDs[newID] = (np.array(iIDs, dtype=np.uint16), MJ.max(), RA.min())
                                
        n_groups = len(set(group_map.ravel())) -1   # kick out LABEL "0"
        return group_map, n_groups, adjoined_pf_map, df_cpfp
    
    
    def group_samples(self, ):        
#        valid_idxs = np.where(self.Base2Ddata>0)
        valid_IDs = np.unique(self.Base2Ddata[(self.Base2Ddata*self.Base2Dwhole) > 0])
        valid_idxs = np.where(np.isin(self.Base2Ddata, valid_IDs))
        
        grps = pd.DataFrame({'idxs': self.Base2Ddata[valid_idxs]}).groupby('idxs')
        
        Mpt_dict = {}
        for ID in valid_IDs: #self.features.index:
            Mpts = self.select_pts(valid_idxs, grps.indices[ID], ID)
            if Mpts.size > 0:
                Mpt_dict.update({ID: Mpts})
        if Mpt_dict.__len__()>0:
            keys = list(Mpt_dict.keys())
            polys = gpd.GeoSeries([geometry.MultiPoint(Mpt_dict[iMpt]) for iMpt in keys], index=keys, name='pf')
        else:
            polys = -1
#        polys = gpd.GeoSeries([geometry.MultiPoint(self.select_pts(valid_idxs, grps.indices[ID], ID)) for ID in self.features.index], index=self.features.index, name='pf')
        return polys
    
    
    def check_edgePts(self, ID, Mpts, locs):
        idxs = np.where(self.obs_edge[locs]==1)[0]
        if ((float(idxs.size)/locs[0].size) >= 0.5) and (self.features.loc[ID, 'area'] < 1000):
            Mpts = np.empty(0)
        else:
            Mpts = Mpts[np.where(self.obs_edge[locs]==0)]
        return Mpts
    
    
    def select_pts(self, main_idxs, sub_idxs, ID):
        locs = (main_idxs[0][sub_idxs], main_idxs[1][sub_idxs])
        Mpts = np.array([self.locX[locs], self.locY[locs]]).T
        if self.features.loc[ID, 'area'] >= 1000:            
            Mpts, locs = self.find_endpoints(Mpts, locs, self.features.loc[ID, 'orient'])
        Mpts = self.check_edgePts(ID, Mpts, locs)
        return Mpts
    
    
    def find_endpoints(self, Mpts, locs, orirad):
        polyA = gpd.GeoDataFrame({'locs':[1], 'geometry':[geometry.MultiPoint(Mpts)]})
        polyB = gpd.GeoDataFrame({'locs':[2], 'geometry':[polyA.convex_hull.buffer(-0.05)[0]]})
        Mpt_edge = gpd.overlay(df1=polyA, df2=polyB, how='difference').explode()
        
        polyX = gpd.GeoSeries([polyA.representative_point()[0]]+[_[1] for _ in Mpt_edge.to_numpy()])
        dists = polyX.distance(polyX[0])[1:].reset_index(drop=True)
        
        X = Mpt_edge.rotate(angle=-orirad, origin=polyA.centroid[0], use_radians=False).x.values
        
        idxs_L = np.where(X<polyA.centroid[0].x)[0]
        idxs_R = np.where(X>polyA.centroid[0].x)[0]
        
        idxs = np.hstack((idxs_L[np.where(dists[idxs_L] > dists[idxs_L].quantile(0.75))],
                          idxs_R[np.where(dists[idxs_R] > dists[idxs_R].quantile(0.75))]))
        
        del polyA, polyB, Mpt_edge, X, dists, idxs_L, idxs_R
#        fig, ax = plt.subplots(figsize=(6, 6))
#        polyA.plot(ax=ax, color='g')
#        polyX[idxs+1].plot( ax=ax, color='k')
        
        ## find the selected point locations
        resPts = gpd.sjoin(polyX[idxs+1].to_frame(name='geometry'), 
                           gpd.GeoDataFrame({'X':locs[0], 'Y':locs[1], 'geometry':[geometry.Point(_) for _ in Mpts]}), 
                           how="inner", op='intersects')
        return Mpts[resPts['index_right']], (resPts['X'].values, resPts['Y'].values)
    
    
    def select_pts0(self, main_idxs, sub_idxs, ID):
        idxs = (main_idxs[0][sub_idxs], main_idxs[1][sub_idxs])
        return np.array([self.locX[idxs], self.locY[idxs]]).T
    
    
    def calc_vec(self, start_pt, end_pt):
        return np.array([end_pt.xy[0][0] - start_pt.xy[0][0], end_pt.xy[1][0] - start_pt.xy[1][0]])

    
    def calc_intersection_angle(self, vec1, vec2):
    #    vec1 = np.array([1, 0])  # sample
    #    vec2 = np.array([-1, 1])  # sample
    #    print(vec1, vec2)
        cos_val = max(min(vec1.dot(vec2) / (np.sqrt(vec1.dot(vec1)) * np.sqrt(vec2.dot(vec2))), 1.), -1)
        angle = np.arccos(cos_val)
        if (angle > self.angle_right) and (vec2[0]<0 and vec2[1]<0): angle *= -1
        return angle
    
    
    def check_union(self, pt0, pt1, pt2, reference_angle=np.deg2rad(90)):
        vec1, vec2 = self.calc_vec(pt0, pt1), self.calc_vec(pt0, pt2)
        angle = np.abs(self.calc_intersection_angle(vec1, vec2))
#        if angle > 1.5*np.pi: angle = 2*np.pi - angle
        if np.abs(angle)> self.angle_right: angle = np.pi - abs(angle)
#        res = True if np.abs(angle) > reference_angle else False
    #    print(res, angle)
        return angle  # , res
    
    
    def cut_cluster(self, TFmap, datamap):
        idxs = np.where(TFmap)
        x_rngs, y_rngs = (max(0, idxs[0].min()-self.save_blank), min(TFmap.shape[0], idxs[0].max()+self.save_blank+1)), (max(0, idxs[1].min()-self.save_blank), min(TFmap.shape[1], idxs[1].max()+self.save_blank+1))
        clus = np.dstack((datamap[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T, 
                            self.raintype[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T,
                            self.rainfall[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T,
                            self.surfdBZ[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T,
                            self.locX[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T, 
                            self.locY[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T  )
                         ).T.astype(np.float32)
        if True in np.isnan(clus):
            clus[np.isnan(clus)] = 0.
        return clus
        
    '''
    def set_psudo_points(self, polys, blank_polys):
        candidate_clus = set(self.ID_arr[self.Mdist.max(axis=0) > 0]) | set(self.ID_arr[self.Mdist.max(axis=1) > 0])
        
        IDs = list(candidate_clus & set(self.features.query('axis_ratio==1').index)) # smaller clusters
        sub_poly = gpd.GeoDataFrame({'ID': IDs, 'geometry': gpd.GeoSeries([polys[iclu].buffer(self.dist_gap) for iclu in IDs])  })   #  
        
        IDs = list(candidate_clus & set(self.features.query('axis_ratio==1').index))  # general clusters
        res = gpd.sjoin(left_df=sub_poly, right_df=blank_polys, op='intersects')
        
        
        return res
    
    
    def group_blank_polys(self, ):
        blank_idxs = np.where(self.Base2Ddata==0)
        gs = {'X': blank_idxs[0], 'Y': blank_idxs[1], 'geometry': gpd.points_from_xy(x=self.locX[blank_idxs], y=self.locY[blank_idxs])}        
        polys = gpd.GeoDataFrame(gs)
        del gs, blank_idxs
        return polys
    
    
    def calc_intersect_pt(self, ipts):
        xy1 = [self.features['center_X'].iloc[ipts[0]], self.features['center_Y'].iloc[ipts[0]]]
        xy2 = [self.features['center_X'].iloc[ipts[1]], self.features['center_Y'].iloc[ipts[1]]]
        kk = [np.tan(np.deg2rad(self.features['oris'].iloc[ipts[0]])), 
              np.tan(np.deg2rad(self.features['oris'].iloc[ipts[1]]))]
        
        x0 = (kk[0]*xy1[0]-kk[1]*xy2[0]-xy1[1]+xy2[1])/(kk[0] - kk[1])
        y0 = kk[0]*(x0-xy1[0]) + xy1[1] 
        return geometry.Point([x0, y0])
    
    
    def calc_center(self, pt1, pt2):
        return geometry.Point([(pt1.xy[0][0] + pt2.xy[0][0])/2, (pt1.xy[1][0] + pt2.xy[1][0])/2])
    
    
    def calc_active_ranges(self, ipts):
        res, angle_window = [], []
        for idx, ipt in enumerate(ipts): 
#            cent = np.array([self.features['center_X'].iloc[ipt], self.features['center_Y'].iloc[ipt]])
            h_angle = self.angle_max*np.sqrt(self.features['axis_ratio'].iloc[ipt])
#            ang0, ang1 = np.deg2rad(self.features['oris'].iloc[ipt])-h_angle-idx*np.pi, np.deg2rad(self.features['oris'].iloc[ipt])+h_angle-idx*np.pi
    #        print(ang0, ang1)
#            a00 = cent+np.array([0.5*self.features['major'].iloc[ipt]*np.cos(ang0), 0.5*self.features['major'].iloc[ipt]*np.sin(ang0)])
    #        a01 = cent-np.array([0.5*self.features['major'].iloc[ipt]*np.cos(ang0), 0.5*self.features['major'].iloc[ipt]*np.sin(ang0)])
#            a10 = cent+np.array([0.5*self.features['major'].iloc[ipt]*np.cos(ang1), 0.5*self.features['major'].iloc[ipt]*np.sin(ang1)])
    #        a11 = cent-np.array([0.5*self.features['major'].iloc[ipt]*np.cos(ang1), 0.5*self.features['major'].iloc[ipt]*np.sin(ang1)])
    #        print(a0, a1)
#            res.extend([geometry.LineString([a00, cent]), 
#                        geometry.LineString([a10, cent])
#                       ])
            
            angle_window.extend([h_angle])
    
        return res, angle_window
    
    def check_angle_involved(self, angles, ipts):
        res, flag = True, False
        area_ratio = self.features['area'].iloc[ipts[1]] / self.features['area'].iloc[ipts[0]]
        if area_ratio <= 0.1: flag=True
        
        for idx, angle in enumerate(angles):
    #        print(ang_ranges[idx][0], angle, ang_ranges[idx][1])
            if abs(angle) > self.angle_right: angle = np.pi - abs(angle)
            if (idx > 0) and flag: 
                self.features.loc[ipts[idx], 'angle_window'] = np.pi
                continue
            if angle > self.features.loc[ipts[idx], 'angle_window']:
                res = False
                break
        return res
    '''
#########################################################################


class Variable_calculations():
    def __init__(self, dadict, platform='TRMM', deal_TMI=False) :
        self.dadict = dadict
        self.platform = platform
        self.deal_TMI = deal_TMI
        
        self.label_mask = generate_binary_structure(2,2)  # for PF generation
        self.pf_ku_basevar = 'nearSurfRain'
        
        if self.dadict['date'] <= '20010823':
            self.pixel_area = param.pixel_area['TRMM_a']
        else:
            self.pixel_area = param.pixel_area[self.platform]
        
        self.Htop_varlist, self.Npxl_varlist = [], []
        self.methods_dBZ = ['max', 'quantile']
        self.quantile_dBZ = 0.95
        self.methods_TBs = ['min', 'quantile']
        self.quantile_TBs = 0.05
        return 
    
    
    def variable_handler_A(self, ):
        self.check_LocalTime()
        self.check_land_ocean()
        self.check_raintype()
        self.check_surfRainPhase()
        self.check_edge()
        return self.dadict
    
    
    def variable_handler_B(self, ):
        self.check_dBZ_height()
        self.check_dBZ_columns()
        ## Precipitation Features
        self.get_PFs()
        
        return self.dadict
    
    
    def remap_2D(self, dataSeq, locs, BasicMap):        
        res = np.zeros_like(BasicMap, dtype=param.fdecimal)
        
        grps = pd.DataFrame({'X': locs[0], 'Y': locs[1], 'idxs': BasicMap[locs]})
        tmp = pd.merge(grps, dataSeq, left_on='idxs', right_index=True)        
        res[tmp['X'].values, tmp['Y'].values] = tmp[dataSeq.columns[0]].values
        return res
    
    
    def func_groupby(self, method_list, var_map, var_value, group_seq, quantile=0.95, remap=False, dtype=np.float32):
        if isinstance(method_list, str): method_list = list(method_list)
        res = {}
        VAR = var_value.astype(np.float32).copy()
        VAR[VAR<=0] = np.nan
        locs = np.where(var_map > 0)
        tmp = pd.DataFrame(np.vstack((var_map[locs], VAR[locs])).T, columns=['idxs','grdvar']).groupby('idxs')
        for idx, method in enumerate(method_list):
            if method in ['sum']:
                ftmp = tmp.sum()
            elif method in ['max']:
                ftmp = tmp.max()
            elif method in ['min']:
                ftmp = tmp.min()
            elif method in ['count']:
                ftmp = tmp.count()
            elif method in ['mean']:
                ftmp = tmp.mean()
            elif method in ['median']:
                ftmp = tmp.median()
            elif method in ['quantile']:
                ftmp = tmp.quantile(q=quantile)
            elif method in ['std']:
                ftmp = tmp.std()
            elif method in ['var']:
                ftmp = tmp.var()
            else:
                print(f'Method error in func_groupby: {method}')
                return -1
                
            ftmp = ftmp.fillna(0).astype(dtype)
            if method in ['quantile']:
                newname = method+str(quantile)[2:]
            else:
                newname = method
            if remap: 
                res[newname] = self.remap_2D(ftmp, locs, var_map)  
            else:
                res[newname] = np.squeeze(ftmp.values)
        del VAR, tmp, ftmp, locs
        return res
    
    
    def calc_var_contribution(self, var_numerator, var_denominator, group_seq, method='count'):
        if isinstance(var_numerator, list):
            numerator = self.func_groupby([method], self.dadict[var_denominator], self.dadict[var_numerator[0]], group_seq)[method]
            denominator = self.func_groupby([method], self.dadict[var_denominator], self.dadict[var_numerator[1]], group_seq)[method]
        else:
            if isinstance(var_numerator, str):
                numerator = self.func_groupby([method], self.dadict[var_denominator], self.dadict[var_numerator], group_seq)[method]
            else:
                numerator = self.func_groupby([method], self.dadict[var_denominator], var_numerator, group_seq)[method]
            denominator = self.func_groupby([method], self.dadict[var_denominator], self.dadict[var_denominator], group_seq)[method]
        res = np.true_divide(numerator, denominator, where=denominator>0)        
        return self.remap_2D(pd.DataFrame({'pf_val':res}, index=group_seq), np.where(self.dadict[var_denominator]>0), self.dadict[var_denominator]) 
    
    
    def calculation_process_A(self, methods, AIM_var, grpMap_var, group_seq, remap=False, quantile=0.95):
        ### calcluate with pandas method
        if AIM_var in self.dadict.keys():                
            res = self.func_groupby(methods, self.dadict[grpMap_var], self.dadict[AIM_var], group_seq, remap=True, quantile=quantile)
            keys = set(res.keys())
            for key in keys:
                self.dadict['_'.join([grpMap_var.split('_groups')[0], AIM_var.split('ku_')[-1], key])] = res.pop(key)
        return 
    
    
    def calculation_process_B(self, AIM_var, grpMap_var, group_seq, AIM_var_data={}, method='count'):
        ### calculate VARIABLE area contribution
        if AIM_var_data.__len__() == 0:
            if isinstance(AIM_var, str): 
                aim_var = AIM_var
            else:
                aim_var = AIM_var[0]
            
            if aim_var in self.dadict.keys(): 
                self.dadict['_'.join([grpMap_var.split('_groups')[0], aim_var.split('ku_')[-1], 'percentage'])] = self.calc_var_contribution(AIM_var, grpMap_var, group_seq, method=method)     
        else:
            for key, value in AIM_var_data.items():
                self.dadict['_'.join([grpMap_var.split('_groups')[0], AIM_var.split('ku_')[-1], key, 'percentage'])] = self.calc_var_contribution(value, grpMap_var, group_seq, method=method)   
        return 
    
    
    def Calc_ku_pf_physical_properties(self, grpMap_var, ngrp):
        group_seq = np.arange(ngrp)+1
        for var in self.Htop_varlist:
            self.calculation_process_A(self.methods_dBZ, var, grpMap_var, group_seq, remap=True, quantile=self.quantile_dBZ)
            self.calculation_process_B(var, grpMap_var, group_seq)
            
        for var in self.Npxl_varlist:
            if var in ['ku_dBZ10_Npxl']: continue
            self.calculation_process_A(['sum', 'max', 'quantile'], var, grpMap_var, group_seq, remap=True, quantile=self.quantile_dBZ)
            self.calculation_process_B([var, 'ku_dBZ10_Npxl'], grpMap_var, group_seq, method='sum')
            

        ##  for TM / VIRS characteristics
        if self.deal_TMI and (param.DATA_source[0] in ['TRMM1Z']):
            for var in ['TMI_pct85', 'TMI_pct37', 'VIRS_ch4']:
                self.calculation_process_A(self.methods_TBs, var, grpMap_var, group_seq, remap=True, quantile=self.quantile_TBs)
                if var in param.Vchars.keys():
                    for thh in param.Vchars[var]:
                        self.calculation_process_B(var, grpMap_var, group_seq, AIM_var_data={f'le{thh:.0f}K': (self.dadict[var]<=thh).astype(np.int32)})
        return 
    
    
    def get_PFs(self, ):
        #calculates contiguous areas where reflectivty is higher than the given threshold
    	#assigns every group a unique number, and returns an array where each index is
    	#replaced with the number of the group it belongs to, or a zero if there was no data there
        
        if self.pf_ku_basevar in self.dadict.keys():
            ####### for whole PF
            self.dadict['ku_pf_groups'], ALL_groups = self.PF_estimation_connect_pixels_method(self.dadict[self.pf_ku_basevar]>0)           
            
            (self.dadict['ku_pf_center_Lon'], self.dadict['ku_pf_center_Lat'], \
                self.dadict['ku_pf_axis_major'], self.dadict['ku_pf_axis_minor'], self.dadict['ku_pf_orien']), _ = \
                self.Get_ellipses(Lons='Longitude', Lats='Latitude', Clusters='ku_pf_groups')
            
            self.dadict['ku_pf_area'], self.dadict['ku_pf_Volrain'] = self.calc_params(self.dadict['ku_pf_groups'])
            self.Calc_ku_pf_physical_properties('ku_pf_groups', ALL_groups)
            
            if 'rainType' in self.dadict.keys():
                conv_idxs, strat_idxs = self.CS_filter(self.dadict['ku_pf_groups'], self.dadict['rainType'])
                self.dadict['ku_pf_conv_area'], self.dadict['ku_pf_conv_Volrain'] = self.calc_params(conv_idxs)
                self.dadict['ku_pf_strat_area'], self.dadict['ku_pf_strat_Volrain'] = self.calc_params(strat_idxs)       
                
                ###############################  for single convective, use traditional connected pixels method
                self.dadict['ku_pf_conv_raw_groups'], conv_groups_raw = self.PF_estimation_connect_pixels_method(conv_idxs>0)              
                
                (self.dadict['ku_pf_conv_raw_center_Lon'], self.dadict['ku_pf_conv_raw_center_Lat'], \
                    self.dadict['ku_pf_conv_raw_axis_major'], self.dadict['ku_pf_conv_raw_axis_minor'], self.dadict['ku_pf_conv_raw_orien']), ellipses_raw = \
                    self.Get_ellipses(Lons='Longitude', Lats='Latitude', Clusters='ku_pf_conv_raw_groups', return_ellipse_summary=True)
                    
                self.dadict['ku_pf_conv_raw_area'], self.dadict['ku_pf_conv_raw_Volrain'] = self.calc_params(self.dadict['ku_pf_conv_raw_groups'])
                self.Calc_ku_pf_physical_properties('ku_pf_conv_raw_groups', conv_groups_raw)
                    
                ##############################  for single convective, BUT use revised block magnet method proposed by Dr. Lei Ji, 202104
                ###  refer to Dr. Xu Weixin, use nearSurfZ as base_TF clusters, threshold sets to 30dBZ
                thh_varname = f'ku_pf_nearSurfdBZ{param.BM_dBZ:02.0f}_groups'
                self.dadict[thh_varname], dBZ_groups = self.PF_estimation_connect_pixels_method(self.dadict['nearSurfZ']>=param.BM_dBZ)  
                
                self.dadict['ku_pf_conv_BM_groups'], conv_groups_BM, self.dadict['ku_pf_conv_BM_groups_adjoined'], adjoin_grps, adjoin_IDs, coupled_pf_properties = \
                    self.PF_estimation_block_magnet_method(Lons='Longitude', Lats='Latitude', Clusters='ku_pf_conv_raw_groups', ellipses_raw=ellipses_raw, 
                                                           #  changed, use base_TF = nearSurfZ groups
                                                           Clusters_whole=thh_varname, #'ku_pf_groups'
                                                           )   #, base_TF=conv_idxs>0
                    
                if isinstance(coupled_pf_properties, pd.DataFrame):
                    self.save_cpfp(coupled_pf_properties)
                del coupled_pf_properties
                        
                if isinstance(adjoin_IDs, int) or (adjoin_IDs.__len__() == 0):
                    keys = set(self.dadict.keys())
                    self.dadict['ku_pf_conv_BM_groups_adjoined'] = np.zeros_like(self.dadict['ku_pf_conv_raw_groups'])
                    for var in keys:
                        if 'raw' in var:
                            self.dadict[var.replace('raw', 'BM')] = self.dadict[var]   # if NO clusters should be grouped, just copy the RAW data
                    print(f"[No BM_pf]: {self.platform} | {self.dadict['date']} | {self.dadict['orbitID']}")
                                        
                else:    
                    (self.dadict['ku_pf_conv_BM_center_Lon'], self.dadict['ku_pf_conv_BM_center_Lat'], \
                        self.dadict['ku_pf_conv_BM_axis_major'], self.dadict['ku_pf_conv_BM_axis_minor'], self.dadict['ku_pf_conv_BM_orien']), _ = \
                        self.Get_ellipses(Lons='Longitude', Lats='Latitude', Clusters='ku_pf_conv_BM_groups')
                    
                    self.dadict['ku_pf_conv_BM_area'], self.dadict['ku_pf_conv_BM_Volrain'] = self.calc_params(self.dadict['ku_pf_conv_BM_groups'])
                    self.Calc_ku_pf_physical_properties('ku_pf_conv_BM_groups', conv_groups_BM)
                    
                    ## for connected clusters part                    
                    adjoin_IDs = self.arrange_df(adjoin_IDs)
                    self.save_grps(adjoin_grps, adjoinDF=adjoin_IDs, subdir=param.fn_adjoin_PFgrps)
#                    if self.platform in ['TRMM', 'GPM']: 
#                    self.plot_BM_groups(adjoin_grps, pdIDs=adjoin_IDs)
                    del adjoin_grps, adjoin_IDs                        
        return 
    
    
    def PF_estimation_connect_pixels_method(self, base_TF):
        return label(base_TF, structure = self.label_mask) 
    
    
    def PF_estimation_block_magnet_method(self, Lons, Lats, Clusters, Clusters_whole, ellipses_raw):        
        ellipses_raw.columns = ['centerLon','centerLat','major','minor','orient']              
        area = self.extract_pfvar(self.dadict[Clusters], self.dadict['ku_pf_conv_raw_area'], varname='area')
        ellipses_raw = pd.merge(ellipses_raw, area, left_index=True, right_index=True)
        del area
        
        grp_instance = Group_block_magnet(self.dadict[Clusters], self.dadict[Lons], self.dadict[Lats], ellipses_raw, self.pixel_area,
                                          self.dadict['rainType'], self.dadict['nearSurfRain'], self.dadict['nearSurfZ'],
                                          self.dadict[Clusters_whole], self.dadict['obs_edge'])
        pf_groups, n_groups, adjoined_pf_grp, adjoin_grps, adjoin_IDs, coupled_pf_properties = grp_instance.arrange_data()
        return pf_groups, n_groups, adjoined_pf_grp, adjoin_grps, adjoin_IDs, coupled_pf_properties
    
    
    def arrange_df(self, indict):
        values = np.array(list(indict.values()), dtype=object)
        res = pd.DataFrame({'BM_ID': indict.keys(), 'contain_IDs_raw': values[:,0], 'Num_IDs_raw': [item.size for item in values[:,0]], 
                            'major_raw_max': values[:,1], 'axis_ratio_raw_min': values[:,2]}, 
                           index=np.full(indict.keys().__len__(), fill_value=self.dadict['orbitID']))
        LocalHour = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], np.tile(self.dadict['LocalHour'],(self.dadict['ku_pf_conv_BM_groups'].shape[1], 1)).T, varname='LocalHour').reset_index().groupby('ID').median()
        major = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_axis_major'], varname='major')
        minor = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_axis_minor'], varname='minor')
        area = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_area'], varname='area')
        
        dBZ20_Max = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ20_Htop_max'], varname='dBZ20_Htop_max')
        dBZ20_Q95 = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ20_Htop_quantile95'], varname='dBZ20_Htop_Q95')
        dBZ20_dep_max = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ20_Npxl_max'], varname='dBZ20_Npxl_max')
        dBZ20_dep_Q95 = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ20_Npxl_quantile95'], varname='dBZ20_Npxl_Q95')
        
        dBZ35_Max = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ35_Htop_max'], varname='dBZ35_Htop_max')
        dBZ35_Q95 = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ35_Htop_quantile95'], varname='dBZ35_Htop_Q95')
        dBZ35_dep_max = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ35_Npxl_max'], varname='dBZ35_Npxl_max')
        dBZ35_dep_Q95 = self.extract_pfvar(self.dadict['ku_pf_conv_BM_groups'], self.dadict['ku_pf_conv_BM_dBZ35_Npxl_quantile95'], varname='dBZ35_Npxl_Q95')        
    
        for var in [LocalHour, major, minor, area, dBZ20_Max, dBZ20_Q95, dBZ20_dep_max, dBZ20_dep_Q95, dBZ35_Max, dBZ35_Q95, dBZ35_dep_max, dBZ35_dep_Q95]:
            res = pd.merge(res, var, left_on='BM_ID', right_index=True)
            if var.columns[0] in ['minor']:
                res['axis_ratio'] = res['minor'] / res['major']
                
#        self.select_pfs(area, name='area', baseMap='ku_pf_conv_BM_groups')   # extra function for cut suitable dataset.
        
        del LocalHour, major, minor, area, dBZ20_Max, dBZ20_Q95, dBZ35_Max, dBZ35_Q95, dBZ20_dep_max, dBZ20_dep_Q95, dBZ35_dep_max, dBZ35_dep_Q95
        return res
    
    
    def extract_pfvar(self, IDdata, vardata, varname):
        locs = np.where(IDdata > 0)
        res = pd.DataFrame({'ID': IDdata[locs], varname: vardata[locs]}).drop_duplicates(keep='first', ignore_index=True).set_index('ID')
        return res
    
    
    def Get_ellipses(self, Lons, Lats, Clusters, return_ellipse_summary=False):
        locs = np.where(self.dadict[Clusters]>0)
        tmp = pd.DataFrame({'X': self.dadict[Lons][locs], 'Y': self.dadict[Lats][locs], 'idxs': self.dadict[Clusters][locs]}).groupby('idxs')
        
        ellipses = []
        clus = np.array(sorted(list(tmp.groups.keys())))
        for iclu in clus: 
#            if iclu == 142:
#                print('**')
            elli_instance = Ellipse_type(lons=tmp.get_group(iclu)['X'].to_numpy(dtype=np.float32), lats=tmp.get_group(iclu)['Y'].to_numpy(dtype=np.float32))
            ellipses.append(elli_instance.fit_ellipse())
            del elli_instance
                        
        dataset = pd.DataFrame({'X': locs[0], 'Y': locs[1], 'idxs': self.dadict[Clusters][locs]})
        pd_ellipses = pd.DataFrame(np.array(ellipses), index=clus, columns=[0,1,2,3,4]) # ['centerLon','centerLat','major','minor','orient']
        dataset = pd.merge(dataset, pd_ellipses, left_on='idxs', right_index=True)
        
        if not return_ellipse_summary: pd_ellipses = None
        
        res = np.zeros((5, self.dadict[Clusters].shape[0], self.dadict[Clusters].shape[1]), dtype=param.fdecimal)
        for ii in range(5):
            res[ii, dataset['X'].values, dataset['Y'].values] = dataset[ii].values
        del tmp, dataset, ellipses, clus
        return res[:], pd_ellipses   #[res[ii] for ii in range(5)]
    
    
    def calc_params(self, pf_groups):
        locs = np.where(pf_groups>0)
        tmp = pd.DataFrame({'idxs': pf_groups[locs], 'RR': self.dadict['nearSurfRain'][locs]}).groupby('idxs')
        
        area = self.remap_2D(self.pixel_area*tmp.count(), locs, pf_groups) 
        volrain = self.remap_2D(self.pixel_area*tmp.sum(), locs, pf_groups) 
        del tmp, locs
        return area, volrain
    
    
    def CS_filter(self, pf_groups, CS_classify):
        conv_idxs = pf_groups*(CS_classify == param.raintypes['convective'])
        strat_idxs = pf_groups*(CS_classify == param.raintypes['stratiform'])
        return conv_idxs, strat_idxs
    
    
    def check_LocalTime(self, ):
        varname = 'LocalHour'
        if 'Minute' in self.dadict.keys():            
            self.dadict[varname] = ((self.dadict['Hour']+self.dadict['Minute']/60. + 
                                    self.dadict['Longitude'][:,int(self.dadict['Longitude'].shape[1]/2)]*24/360)%24).astype(np.float32)
            self.dadict.pop('Minute') 
        elif 'Hour' in self.dadict.keys():
            self.dadict[varname] = ((self.dadict['Hour'] + 
                                    self.dadict['Longitude'][:,int(self.dadict['Longitude'].shape[1]/2)]*24/360)%24).astype(np.float32)    
            
        idxs = np.where(self.dadict[varname]>=24)
        self.dadict[varname][idxs] -= 24
        idxs = np.where(self.dadict[varname]<0)
        self.dadict[varname][idxs] += 24
        
        varname = 'LocalHour_tmi'    
        flag_exe = False
        if 'Minute_tmi' in self.dadict.keys():            
            self.dadict[varname] = ((self.dadict['Hour_tmi']+self.dadict['Minute_tmi']/60. + 
                                    self.dadict['TMI_Lon'][:,int(self.dadict['TMI_Lon'].shape[1]/2)]*24/360)%24).astype(np.float32)
            self.dadict.pop('Minute_tmi')  
            flag_exe = True
        elif 'Hour_tmi' in self.dadict.keys():            
            self.dadict[varname] = ((self.dadict['Hour_tmi'] + 
                                    self.dadict['TMI_Lon'][:,int(self.dadict['TMI_Lon'].shape[1]/2)]*24/360)%24).astype(np.float32)
            flag_exe = True
            
        if flag_exe:
            idxs = np.where(self.dadict[varname]>=24)
            self.dadict[varname][idxs] -= 24
            idxs = np.where(self.dadict[varname]<0)
            self.dadict[varname][idxs] += 24
        return 
    
    
    def check_land_ocean(self,):
        if 'TMI_LandOcean19' in self.dadict.keys():  # page 1295, filespec.GPM.pdf
            varname = 'TMI_LandOcean19'
            self.dadict['LandOcean'] = np.full_like(self.dadict[varname], fill_value=param.landocean['unknown'], dtype=np.int32)
            
            for val, flg in zip((1,2,12), (param.landocean['ocean'], param.landocean['SeaIce'], param.landocean['inLandLake'])):
                idxs = np.where(self.dadict[varname]==val)
                self.dadict['LandOcean'][idxs] = flg

            idxs = np.where(np.logical_and(self.dadict[varname]>=3,self.dadict[varname]<=11))
            self.dadict['LandOcean'][idxs] = param.landocean['land']            
            idxs = np.where(np.logical_and(self.dadict[varname]>=13, self.dadict[varname]<=15))
            self.dadict['LandOcean'][idxs] = param.landocean['coast']
            
        elif 'NS/PRE/landSurfaceType' in self.dadict.keys():
            varname = 'NS/PRE/landSurfaceType'
            self.dadict['LandOcean'] = np.full_like(self.dadict[varname], fill_value=param.landocean['unknown'], dtype=np.int32)
            self.dadict['LandOcean'] = self.dadict[varname] // 100
            
        elif 'TMI_LandOcean09' in self.dadict.keys():
            varname = 'TMI_LandOcean09'
            valid_idxs = np.where(np.logical_and(0<self.dadict[varname], self.dadict[varname]<=30))            
            self.dadict['LandOcean'] = np.full_like(self.dadict[varname], fill_value=param.landocean['unknown'], dtype=np.int32)
            self.dadict['LandOcean'][valid_idxs] = self.dadict[varname][valid_idxs] // 10 -1
        
        elif 'status' in self.dadict.keys():
            varname = 'status'
            valid_idxs = np.where(np.logical_and(self.dadict[varname]//100 == 0, self.dadict[varname]>=0))
            if valid_idxs[0].size > 0 :
                self.dadict['LandOcean'] = np.full_like(self.dadict[varname], fill_value=param.landocean['unknown'], dtype=np.int32)
                self.dadict['LandOcean'][valid_idxs] = self.dadict[varname][valid_idxs] % 10
              
        else:
            pass
                    
        self.dadict.pop(varname)     
        return 


    def check_raintype(self, ):
        for varname, ratio in zip(('NS/CSF/typePrecip', 'PR__RAINTYPE', 'rainType', 'shallowRain'),(10000000, 10000000, 100, 10)):
            if varname in self.dadict.keys(): 
                valid_idxs = np.where(self.dadict[varname]>0)
                self.dadict[varname][valid_idxs] = (self.dadict[varname][valid_idxs]//ratio)
                if varname in ['NS/CSF/typePrecip', 'PR__RAINTYPE']:
                    self.dadict['rainType'] = np.zeros_like(self.dadict[varname], dtype=np.int32)
                    self.dadict['rainType'] = self.dadict[varname].astype(np.int32)
                    self.dadict.pop(varname)
                else:
                    self.dadict[varname] = self.dadict[varname].astype(np.int32)
                break
        return 
    
    
    def check_surfRainPhase(self, ):
        if 'PR__PHASE' in self.dadict.keys():
            varname = 'PR__PHASE'            
        elif 'NS/SLV/phaseNearSurface' in self.dadict.keys():
            varname = 'NS/SLV/phaseNearSurface'
            
        valid_idxs = np.where(self.dadict[varname]<255)
        self.dadict['surfRainPhase'] = np.full_like(self.dadict[varname], fill_value=-999, dtype=np.int32)
        self.dadict['surfRainPhase'][valid_idxs] = self.dadict[varname][valid_idxs] / 100
        self.dadict.pop(varname) 
        self.assign_nan(self.dadict, ['surfRainPhase'], has_thh=False)        
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
    
    
    '''
    def correct_dBZ_profile(self, ):               
        if 'scLocalZenith' in self.dadict.keys(): 
            self.dadict['scLocalZenith'] = np.tan(np.deg2rad(self.dadict['scLocalZenith'])).astype(param.fdecimal)
        
        
        return 
    '''
    
    def check_dBZ_columns(self, ):
        if 'dBZ_profile' in self.dadict.keys():
            varname = 'dBZ_profile'
#            platform = 'GPM'  #varname.split('_')[0]
            for dBZ in param.lev_dBZ+[10.]:
                newvar = f'ku_dBZ{dBZ:.0f}_Npxl'
                self.Npxl_varlist.append(newvar)
                self.dadict[newvar] = np.zeros_like(self.dadict[varname][:,:,0])
                self.dadict[newvar] = self.dadict[newvar].astype(np.float32)
                
                idxs = np.where(self.dadict[varname] >= dBZ)
                if idxs[0].size > 0:
                    tmp = pd.DataFrame(dict(zip(('X','Y','Z0'), idxs)))
                    NLoc = tmp.groupby(by=(['X','Y']), as_index=False).count()
                    del tmp
                     
                    self.dadict[newvar][NLoc['X'].values, NLoc['Y'].values] = NLoc['Z0'].to_numpy().astype(np.float32)
                    del NLoc        
        return
    
    
    def check_dBZ_height(self, ):
        if 'dBZ_profile' in self.dadict.keys():
            varname = 'dBZ_profile'
            HGT = pd.DataFrame({'HGT': param.zdim[self.platform]['HGT']})
            for dBZ in param.lev_dBZ:
                newvar = f'ku_dBZ{dBZ:.0f}_Htop'
                self.Htop_varlist.append(newvar)
                self.dadict[newvar] = np.zeros_like(self.dadict[varname][:,:,0])  #, fill_value=param.MeaningLess_val['none']
                
                idxs = np.where(self.dadict[varname] >= dBZ)
                if idxs[0].size > 0:
                    tmp = pd.DataFrame(dict(zip(('X','Y','Z0'), idxs)))
                    TopLoc = tmp.groupby(by=(['X','Y']), as_index=False).min()
                    del tmp
                    TopLoc = pd.merge(TopLoc, HGT, left_on='Z0', right_index=True)
                    TopLoc['dBZ0'] = self.dadict[varname][(TopLoc['X'].values, TopLoc['Y'].values, TopLoc['Z0'].values)]
                    TopLoc['dBZ1'] = self.dadict[varname][(TopLoc['X'].values, TopLoc['Y'].values, TopLoc['Z0'].values-1)]
                    TopLoc.fillna(value=17, inplace=True)
                    
                    self.dadict[newvar][TopLoc['X'].values, TopLoc['Y'].values] = self.Interp_Hgt(TopLoc, ['HGT', 'dBZ0', 'dBZ1'], dBZ, param.zdim[self.platform]['dH'])
                    del TopLoc
        return 
    
    
    def Interp_Hgt(self, DFdata, varname, thh, interval):
        return (DFdata[varname[0]] + interval*(DFdata[varname[1]] - thh) / (DFdata[varname[1]] - DFdata[varname[2]])).astype(param.fdecimal)
    
    
    def check_edge(self, ):
        self.dadict['obs_edge'] = np.zeros_like(self.dadict['rainType'], dtype=np.int32)
        self.dadict['obs_edge'][:,[0,-1]] = 1
        return 
        
    
    def save_grps(self, dadict, adjoinDF=-1, mode='a', subdir=''):
        sfn = os.path.join(param.res_Disk, param.op_res, param.op_data, self.platform, subdir, '_'.join([self.platform, subdir, self.dadict['date'][:4]]))
        store_pixel = h5py.File(sfn+'.h5', mode)
        key = '_'.join([self.dadict['date'], self.dadict['orbitID']])
        if key in store_pixel.keys(): 
            grp = store_pixel[key]    
        else:
            grp = store_pixel.create_group(key)
                
        for key in set(dadict.keys()):
            if key not in grp.keys():
                grp.create_dataset(key, dadict[key].shape, data=dadict[key], compression="gzip", compression_opts=9, dtype=dadict[key].dtype)            
        store_pixel.close()        
        
        if not isinstance(adjoinDF, int):
            adjoinDF.insert(0, 'orbitID', adjoinDF.index) 
            if os.path.exists(sfn+'_IDs.csv'):
                adjoinDF.to_csv(sfn+'_IDs.csv', mode='a', header=False, index=False)
            else:
                adjoinDF.to_csv(sfn+'_IDs.csv', header=True, index=False)
        return
    
    
    def save_cpfp(self, df):
        if df.size > 0:
            sfn = os.path.join(param.res_Disk, param.op_res, param.op_data, self.platform, param.fn_Conv_adjoin_table, #'month', 
                               '_'.join([self.platform, param.fn_Conv_adjoin_table, self.dadict['date'][:-2]+'.csv']))
            df.insert(0, 'orbitID', np.full(df.index.size, self.dadict['orbitID']))
            if os.path.exists(sfn):
                df.to_csv(sfn, mode='a', header=False, index=False)
            else:
                df.to_csv(sfn, header=True, index=False)
        return
    
    
    def __check_DIR__(self, DIR):
        if not os.path.exists(DIR): os.makedirs(DIR)
        return
    
    
    def data_load(self, fn='', keylist={}):
        res = {}        
        data = h5py.File(fn, 'r')        # refer to writing method01
        for key in data.keys():
            res[key] = {}
            res.update({item for item in data[key].items()})              
        data.close()
        return res
    
    #################################
    def plot_BM_groups(self, dadict, pdIDs):   
        pdIDs.index = pdIDs['BM_ID']
        for key in set(dadict.keys()):
            idx = np.where(self.dadict['ku_pf_conv_raw_groups']==int(key))
            ellipseB = [[(self.dadict['ku_pf_conv_BM_center_Lon'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_BM_center_Lat'][idx[0][0], idx[1][0]]),
                             (self.dadict['ku_pf_conv_BM_axis_major'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_BM_axis_minor'][idx[0][0], idx[1][0]]),
                              self.dadict['ku_pf_conv_BM_orien'][idx[0][0], idx[1][0]]]]
            axRatio = [f'{ellipseB[0][1][1]/ellipseB[0][1][0]: .3f}'.lstrip()]
            
            ellipseA = [[(self.dadict['ku_pf_conv_raw_center_Lon'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_raw_center_Lat'][idx[0][0], idx[1][0]]),
                                 (self.dadict['ku_pf_conv_raw_axis_major'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_raw_axis_minor'][idx[0][0], idx[1][0]]),
                                  self.dadict['ku_pf_conv_raw_orien'][idx[0][0], idx[1][0]]]]
            axRatio.append(f'{ellipseA[0][1][1]/ellipseA[0][1][0]: .3f}\n'.lstrip())
            
            for ikey in pdIDs.loc[int(key), 'contain_IDs_raw']:
                if ikey == int(key): continue
                idx = np.where(self.dadict['ku_pf_conv_raw_groups']==ikey)
                ellipseA.append([(self.dadict['ku_pf_conv_raw_center_Lon'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_raw_center_Lat'][idx[0][0], idx[1][0]]),
                                 (self.dadict['ku_pf_conv_raw_axis_major'][idx[0][0], idx[1][0]], self.dadict['ku_pf_conv_raw_axis_minor'][idx[0][0], idx[1][0]]),
                                  self.dadict['ku_pf_conv_raw_orien'][idx[0][0], idx[1][0]]])
                axRatio[-1] = axRatio[-1]+(f'{ellipseA[-1][1][1]/(ellipseA[-1][1][0]+param.min_eps): .3f}\n'.lstrip())
                        
            ll_rng = np.around([dadict[key][-2].min()-0.1, dadict[key][-2].max()+0.1, dadict[key][-1].min()-0.1, dadict[key][-1].max()+0.1], 1)
            #### plot parts
            self.plot_CS(dadict[key][-2], dadict[key][-1], dadict[key][1], '_'.join([self.platform,'CS',self.dadict['date'],self.dadict['orbitID'],key]),
                         ckind='cs', ll_rng=ll_rng, lw=1, ellipseA=ellipseA, ellipseB=ellipseB, text=axRatio, pic_subdir=self.dadict['date'][:4])
            self.plot_rain(dadict[key][-2], dadict[key][-1], dadict[key][2], '_'.join([self.platform,'rainfall',self.dadict['date'],self.dadict['orbitID'],key]),
                           save_zero=False, cmap_rng=list(np.arange(6)+1), ckind='pr1',extend='max',ll_rng=ll_rng, lw=1, 
                           ellipseA=ellipseA, ellipseB=ellipseB, text=axRatio, pic_subdir=self.dadict['date'][:4]) 
            self.plot_dBZ(dadict[key][-2], dadict[key][-1], dadict[key][3], '_'.join([self.platform,'nearSurfZ',self.dadict['date'],self.dadict['orbitID'],key]),
                           save_zero=False, cmap_rng=list(np.arange(5)+2), ckind='pr1',extend='max',ll_rng=ll_rng, lw=1, 
                           ellipseA=ellipseA, ellipseB=ellipseB, text=axRatio, pic_subdir=self.dadict['date'][:4]) 
            
            '''
            self.plot_CS(dadict[key][3], dadict[key][4], dadict[key][1], '_'.join([self.platform,'CS',self.dadict['date'],self.dadict['orbitID'],key,'X']),
                         ckind='cs', ll_rng=ll_rng, lw=1, ellipseA=ellipseA, text=[axRatio[-1]])
            self.plot_rain(dadict[key][3], dadict[key][4], dadict[key][2], '_'.join([self.platform,'rainfall',self.dadict['date'],self.dadict['orbitID'],key,'X']),
                           save_zero=False, cmap_rng=list(np.arange(6)+1), ckind='pr1',extend='max',ll_rng=ll_rng, lw=1, 
                           ellipseA=ellipseA, text=[axRatio[-1]])
            '''
        return
    
    
    def plot_CS(self, Xs, Ys, Data2d, name, cmap_rngs=[3], levs=[1,2], category=['S','C'], ll_rng=[], lw=3, 
                c_firstwhite=False, ckind='cs', extend='neither', ellipseA=[], ellipseB=[], text=[], pic_subdir=''):
        plot = Plotting(global_xy=False, cb_shrink=0.4)
        
        plot.plot_grid('Ku_pf_conv_BM', name, Xs, Ys, Data2d, 'RainType', #var_CS, 
                       dpi=600, p_sca=True, marker='.', cmap_rng=[3], 
                       levs=[1,2], size=30, category=['S','C'], 
                       ellipses=ellipseA, ellipses2=ellipseB, ellipses_colors=['r'], 
                       ll_rng=ll_rng, lw=lw, c_firstwhite=False, ckind=ckind, extend=extend, text=text, pic_subdir=pic_subdir)
        del plot        
        return 
    
    
    def plot_rain(self, Xs, Ys, data2D, name, kind='', unit='',save_zero=True, p_sca=False, lw=3, ll_rng=[], 
                       levs=[], cmap_rng=[], extend='neither', ckind='pr', ellipseA=[], ellipseB=[], text=[], pic_subdir=''):
        if levs.__len__() == 0:
            levs = self.rainfall_levs(data2D.max())
            
        tmpdata = data2D.copy()
        if not save_zero:
            tmpdata[tmpdata<=0] = -1
            
        plot = Plotting(global_xy=False, cb_shrink=0.5)

        plot.plot_grid('Ku_pf_conv_BM', name, Xs, Ys, tmpdata, 'rain', unit=r'(mm$\ $h$^{-1}$)',
                       levs=levs, cmap_rng=cmap_rng, extend=extend, ckind=ckind, 
                       ellipses=ellipseA, ellipses2=ellipseB, ellipses_colors=['r'], 
                       dpi=600, ll_rng=ll_rng, lw=lw, text=text, pic_subdir=pic_subdir) #
        del tmpdata, plot 
        return 
    
    
    def plot_dBZ(self, Xs, Ys, data2D, name, kind='', unit='',save_zero=True, p_sca=False, lw=3, ll_rng=[], 
                       levs=[], cmap_rng=[], extend='neither', ckind='pr', ellipseA=[], ellipseB=[], text=[], pic_subdir=''):
        if levs.__len__() == 0:
            levs = np.arange(10, 60, 1)
            
        tmpdata = data2D.copy()
#        if not save_zero:
#            tmpdata[tmpdata<=0] = -1
            
        plot = Plotting(global_xy=False, cb_shrink=0.5)

        plot.plot_grid('Ku_pf_conv_BM', name, Xs, Ys, tmpdata, 'near Surface dBZ', unit='',
                       levs=levs, cmap_rng=cmap_rng, extend=extend, ckind=ckind, pmesh=True,
                       ellipses=ellipseA, ellipses2=ellipseB, ellipses_colors=['r'], 
                       dpi=600, ll_rng=ll_rng, lw=lw, text=text, pic_subdir=pic_subdir) #
        del tmpdata, plot 
        return 
    
    
    def rainfall_levs(self, maxval):
        if maxval <= 5:
            levs = [0.05]+np.arange(0.1, 5.01, 0.1).tolist()
        elif maxval <= 10:
            levs = [0.05, 0.1]+np.arange(0.2, 10.01, 0.2).tolist()
        elif maxval <= 30:
            levs = [0.05, 0.1]+np.arange(0.5, 30.01, 0.5).tolist()
        elif maxval <= 50:
            levs = [0.05, 0.1]+np.arange(1, 30.01, 1).tolist()+np.arange(32, 50.01, 2).tolist()
        elif maxval <= 100:
            levs = [0.05, 0.1]+np.arange(1, 30.01, 1).tolist()+np.arange(32, 50.01, 2).tolist()+np.arange(55, 100.01, 5).tolist()
        elif maxval <= 500:
            levs = [0.05, 0.1, 1]+np.arange(2, 50.01, 2).tolist()+np.arange(55, 150.01, 5).tolist()+np.arange(200, 500.01, 50).tolist()
        else:
            levs = [0.05, 0.1, 1, 2, 5]+np.around(np.linspace(10, maxval+10, 40), 0).tolist()
            
        if 5< maxval <=500:
            return levs[:np.where(levs >= maxval)[0].min()]
        else:
            return levs

###################################################
    def select_pfs(self, selectDF, name='area', baseMap='', thh=500):
        pfIDs = selectDF.query(f'{name}>=@thh').index
        
        res = {}
        for ID in pfIDs:
            TFmap = np.isin(self.dadict[baseMap], ID).astype(np.int32)
            res[str(ID)] = self.cut_pf_group(TFmap, varlist=['nearSurfRain', 'Longitude', 'Latitude'])        
        self.save_grps(res, subdir=param.fn_Ku_pfs[0])
        return


    def cut_pf_group(self, TFmap, varlist):
        idxs = np.where(TFmap)
        x_rngs, y_rngs = (max(0, idxs[0].min()-param.save_blank), min(TFmap.shape[0], idxs[0].max()+param.save_blank+1)), (max(0, idxs[1].min()-param.save_blank), min(TFmap.shape[1], idxs[1].max()+param.save_blank+1))
        clus = np.dstack((TFmap[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T, 
                          (self.dadict[varlist[0]]*TFmap)[x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T, 
                          self.dadict[varlist[1]][x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T,
                          self.dadict[varlist[2]][x_rngs[0]:x_rngs[1], y_rngs[0]:y_rngs[1]].T
                         )).T.astype(np.float32)
        if True in np.isnan(clus):
            clus[np.isnan(clus)] = 0.
        return clus
        
