# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:27:41 2020

@author: Leo_Ji
"""

import module_Parameters as param
from module_selfdef_colormaps import self_def_colormaps as sdcmap

import matplotlib
matplotlib.use('PDF')
matplotlib.rcParams['pdf.compression'] = 9

import matplotlib.pyplot as plt
from matplotlib import cm,colors, ticker
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import copy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader

import os
import numpy as np


class Plotting():
    def __init__(self, data=np.empty([0]), obtID='00000', lon_center=None, global_xy=True, cb_shrink=0.9, fontsize=5):
        self.data_sav = data
        self.icmap = sdcmap()
        self.curr_obtID = obtID
#        cm.register_cmap(cmap=icmap.set_cmap())
        self.XY_ticks_global = global_xy
        self.cb_shrink = cb_shrink
        self.fontsize=fontsize
        
        if lon_center == None:
            self.lon_center = param.plot_lon_center
        else:
            self.lon_center = lon_center
        return
    
    
    def __load_panel_settings__(self):  # , lon_rng=np.array([-180., 180.])
        sns.set_context('paper', font_scale=1)  # 
        
        proj = ccrs.PlateCarree(central_longitude=self.lon_center)  # np.mean(lon_rng)
#        proj = ccrs.Mercator()
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection=proj))
        return fig, ax, proj
    
    
    def __load_ax_settings__(self, fig, ax, proj, obj, var, kind='', rng=[-180, 180, -70, 70], category=[], unit='', is_China=False):
        cb_orientation, pad, aspect ='horizontal', 0.04, 30
#        cb_orientation, pad, aspect ='vertical', 0.02, 14
        cb = fig.colorbar(obj, shrink=self.cb_shrink, aspect=aspect, pad=pad,ax=ax, orientation=cb_orientation) # 
        cb.ax.tick_params(direction='in', length=4)
        if len(category) > 0:
            if cb_orientation in ['', 'vertical']: 
                cb.ax.yaxis.set_major_locator(ticker.FixedLocator([1.25, 1.75]))
                cb.ax.set_yticklabels(category, fontdict={'fontsize': self.fontsize})
            elif cb_orientation in ['horizontal']:
                cb.ax.xaxis.set_major_locator(ticker.FixedLocator([1.25, 1.75]))
                cb.ax.set_xticklabels(category, fontdict={'fontsize': self.fontsize})
#            cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{int(val)}%'))
            
            cb.ax.tick_params(direction='in', length=0)
        else:
            cb.ax.tick_params(direction='in', length=4)
            cb.ax.tick_params(which='minor', direction='in', length=2)
            tick_labs = self.__shrink_tail_zeros__(cb.get_ticks())
            if cb_orientation in ['', 'vertical']: 
                cb.ax.set_yticklabels(tick_labs, fontdict={'fontsize': self.fontsize})
            elif cb_orientation in ['horizontal']:
                cb.ax.set_xticklabels(tick_labs, fontdict={'fontsize': self.fontsize})            
            
#        cb.set_label('Precipitation (mm)')
        if 'count' not in kind: 
            if 'ratio' in kind:
                cb.set_label(f'Grid_ratio ({unit})', fontdict={'fontsize': self.fontsize}, labelpad=1)
            elif 'HGT' in kind:
                cb.set_label(f'Height ({unit})', fontdict={'fontsize': self.fontsize}, labelpad=1)
            elif 'mean' in kind:
                cb.set_label(f'({unit})', fontdict={'fontsize': self.fontsize}, labelpad=1)    
            elif 'rain' in kind:
                cb.set_label(f'Rainfall ({unit})', fontdict={'fontsize': self.fontsize}, labelpad=1)
#            elif var in param.desp_Vars.keys(): 
#                cb.set_label(self.setlabel(param.desp_Vars[var]))
            else:
                cb.set_label(f'{kind}', fontdict={'fontsize': self.fontsize}, labelpad=1)
        
#        ax.coastlines()
        
        # --设置地图属性
        if is_China:
            provinces = cfeature.ShapelyFeature(
                    Reader(param.shp_file).geometries(),
                    proj, edgecolor='k',
                    facecolor='beige'    #'none'
                    )
        map_scale = '50m'
        ax.add_feature(cfeature.LAND.with_scale(map_scale), facecolor='beige', alpha=0.5, zorder=0)####添加陆地######
        ax.add_feature(cfeature.COASTLINE.with_scale(map_scale),lw=0.5,zorder=3)#####添加海岸线#########
        if is_China: ax.add_feature(provinces, lw=0.2, zorder=1)
##        ax.add_feature(cfeature.RIVERS.with_scale(map_scale),lw=0.25)#####添加河流######
##        ax.add_feature(cfeature.LAKES.with_scale(map_scale))######添加湖泊#####
#        ax.add_feature(cfeature.BORDERS.with_scale(map_scale), linestyle='-',lw=0.25)####不推荐，我国丢失了藏南、台湾等领土############
##        ax.add_feature(cfeature.OCEAN.with_scale(map_scale))######添加海洋######## , color='b'
        gl = ax.gridlines(draw_labels=True, linewidth=1, linestyle=':', color='gray', alpha=0.6) #crs=proj, 
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style=gl.ylabel_style={'size':self.fontsize}#修改经纬度字体大小                             
        gl.xpadding = gl.ypadding = 15
#        gl.xlocator._nbins = int(gl.xlocator._nbins/2)    
#        gl.ylocator._nbins = int(gl.ylocator._nbins/2) 
        
#        ax.set_extent(rng, crs=proj) 
              
        '''
        if self.XY_ticks_global:
            ##   for Global
            ax.set_xticks(np.arange(rng[0], rng[1]+1, 60), crs=proj)
            ax.set_yticks(np.arange(-60, 61, 20), crs=proj)
        else:            
            ax.set_xticks(np.arange(rng[0], rng[1]+0.0001, max(round((rng[1]-rng[0])/4,2), 0.5)), crs=proj)
            ax.set_yticks(np.arange(rng[2], rng[3]+0.0001, max(round((rng[3]-rng[2])/4,2), 0.5)), crs=proj)
        '''
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        return
    
    
    def __shrink_tail_zeros__(self, fnums):
        lstr = []
        for num in fnums:
            s = f'{num:.2f}'
            if s[-1]=='0':
                s = f'{num:.1f}'  
                if s[-1]=='0':
                    s = f'{num:.0f}'
            lstr.extend([s])
        return lstr
    
    
    def plot_orbit(self, var, name, orbit=-1, levs=[], ll_rng=[], cmap='rainbow_r', extend='neither'):  # orbit可为list，存放需要绘制的轨道号，整数
        fig, ax, proj = self.__load_panel_settings__()
        
#        cmaps = colors.ListedColormap(param.collev, 'indexed')
#        norm = colors.BoundaryNorm(param.pr_levels, cmaps.N)
#        norm = colors.Normalize(vmin=param.pr_levels[0], vmax=param.pr_levels[-1])
        
        if orbit == -1:
            orbit = self.data_sav.keys()                
        else:
            if isinstance(orbit, int): orbit = [orbit]
            
        for ID in orbit:
            if var in ['tmi.pct85']: self.data_sav[ID][var][self.data_sav[ID][var]>273] = -1
            data = np.ma.array(self.data_sav[ID][var], mask=(self.data_sav[ID][var]<0))
            if var in ['tmi.surfprecip', 'pr.nearSurfRain']: data = np.ma.array(data, mask=(data<0.1))
 #           con = ax.contourf(self.data_sav[ID]['Lon'], self.data_sav[ID]['Lat'], 
 #                             self.data_sav[ID][var], cmap=cmaps, norm=norm, levels=param.pr_levels, extend='max')
#            sca = ax.scatter(self.data_sav[ID]['pr.lon'], self.data_sav[ID]['pr.lat'], 
#                             c=self.data_sav[ID][var], cmap='rainbow', norm=colors.LogNorm(),
#                             marker='.', edgecolors='none', s=2, alpha=1)
            cmaps = copy.copy((cm.get_cmap(cmap)))
            if var in ['tmi.surfaceflag']:
                if 'tmi.' in var:
                    lons, lats = self.data_sav[ID]['tmi.lonhi'], self.data_sav[ID]['tmi.lathi']
                elif 'pr.' in var:
                    lons, lats = self.data_sav[ID]['pr.lon'], self.data_sav[ID]['pr.lat']
                sca = ax.scatter(lons, lats,  
                             c=data, cmap=cmaps, marker='.', edgecolors='none', s=2, alpha=1, zorder=2)
            else:
                cmaps = self.icmap.set_cmap(rng=[15,60], c_firstwhite=False)
                if 'tmi.' in var:
                    lons, lats = self.data_sav[ID]['tmi.lonhi'], self.data_sav[ID]['tmi.lathi']
                elif 'pr.' in var:
                    lons, lats = self.data_sav[ID]['pr.lon'], self.data_sav[ID]['pr.lat']
                sca = ax.contourf(lons, lats, data, norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=cmaps.N),
                             cmap=cmaps, levels=levs, extend=extend, alpha=0.7, zorder=2)  #norm=colors.LogNorm(),
                           #  ,marker='.', edgecolors='none', s=2, alpha=1)
            
            self.__load_ax_settings__(fig, ax, proj, sca, name, rng=ll_rng)
            ax.set_title(var, fontsize=7)
    
            self._save_fig(var+'_orbit_'+self.curr_obtID, fig)
#            fig.savefig(fnn, dpi=300, bbox_inches='tight') 
        return
    
    
    def plot_grid(self, var, idate, grid_x, grid_y, data, method='', levs=[], p_sca=False, marker='s', size=0.1, linewidths=1, category=[],
                  ellipses=[], ellipses2=[], ellipses_colors=[], 
                  polygons=[], polygons2=[], polygon_colors=[],
                  draw_edge=False, ll_rng=[], lw=None, dpi=1200, cmap='', pmesh=False,
                  cmap_rng=[], c_firstwhite=False, extend='neither', ckind='pr', unit='', text=[], pic_subdir=''):        
        fig, ax, proj = self.__load_panel_settings__() # lon_rng=lon_rng
        datap = np.ma.array(data, mask=data<(param.MeaningLess_val['missing']+1), fill_value=param.MeaningLess_val['missing'])
                
        if len(cmap_rng) > 0:
            cmap = self.icmap.set_cmap(rng=cmap_rng, c_firstwhite=c_firstwhite, kind=ckind)            
        else:
            if cmap.__len__() == 0: cmap = copy.copy((cm.get_cmap("bwr")))
        
        if len(levs) == 0:
            if p_sca:
                TF_valid = data>=0
                obj = ax.scatter(grid_x[TF_valid].ravel(), grid_y[TF_valid].ravel(),  
                             c=data[TF_valid].ravel(), cmap=cmap, norm=colors.Normalize(min(data[TF_valid]),max(data[TF_valid]),1),
                             marker=marker, s=size, linewidths=linewidths, alpha=1, zorder=2, rasterized=True)
            else:
                obj = ax.contourf(grid_x, grid_y, datap, levels=49,
                          cmap=cmap, alpha=1, zorder=2, extend=extend) # norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=len(levs)-1), 
                if len(cmap_rng) > 0:
                    if extend in ['min','both']: obj.cmap.set_under('white')
                    if extend in ['max','both']: obj.cmap.set_over('indigo')
        else:
            if p_sca:
                TF_valid = np.isin(data, np.array(levs))
                obj = ax.scatter(grid_x[~TF_valid].ravel(), grid_y[~TF_valid].ravel(),  
                             c='lightgrey', linewidths=linewidths, 
                             marker=marker, s=size*0.1, alpha=1, zorder=1, rasterized=True)
                
                obj = ax.scatter(grid_x[TF_valid].ravel(), grid_y[TF_valid].ravel(),  
                             c=data[TF_valid].ravel(), cmap=cmap, linewidths=0, 
                             marker=marker, s=size, alpha=1, zorder=2, rasterized=True) # edgecolors='none', 
            else:
                if pmesh:                
                    obj0 = ax.pcolormesh(grid_x, grid_y, datap, cmap=cmap, shading='auto', 
                                    norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=cmap.N), 
                                    edgecolors='none', alpha=1, zorder=2)
                    ## for colorbar extend
                    zod, visible = 0, False
                else:
                    zod, visible = 2, True                    
            
                obj = ax.contourf(grid_x, grid_y, datap, visible=visible, 
                          cmap=cmap, levels=levs, norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=cmap.N), 
                          alpha=1, zorder=zod, extend=extend)  # colors=param.collev1
#        im = ax.imshow(data.T, extent=(param.rng_lon[0], param.rng_lon[1], param.rng_lat[0], param.rng_lat[1]), 
#                       origin='upper', cmap=cmap, norm=colors.LogNorm(), alpha=1)
##                obj.norm.clips = False
                if len(cmap_rng) > 0:
                    if extend in ['min','both']: obj.cmap.set_under('white')
                    if extend in ['max','both']: obj.cmap.set_over('indigo')
                
        if draw_edge: self.draw_swath_edges(ax, grid_x, grid_y, data, lw=0.3)            
        if len(ellipses) > 0: self.draw_ellipses(ax, ellipses, lw=lw)   #, colors=ellipses_colors
        if len(ellipses2) > 0: self.draw_ellipses(ax, ellipses2, lw=lw, colors=ellipses_colors)
        if len(polygons) > 0: self.draw_polygon(ax, polygons, lw=lw)
        if len(polygons2) > 0: self.draw_polygon(ax, polygons2, lw=lw, colors=polygon_colors)
            
        self.__load_ax_settings__(fig, ax, proj, obj, var, kind=method, rng=ll_rng, category=category, unit=unit)
#        ax.set_title(idate, fontdict={'fontsize': self.fontsize}, pad=3)
        
        if text.__len__() > 0:
            ax.text(x=0.99, y=0.99, s='Axial Ratio', ma='right', ha='right', va='top', fontsize=self.fontsize, color='black', transform=ax.transAxes)                
            if text.__len__() == 1:
                ax.text(x=0.97, y=0.99-0.03, s=text[0], ma='right', ha='right', va='top', fontsize=self.fontsize, color='black', transform=ax.transAxes)
            else:
                ii=1
                for val, color in zip(text, (ellipses_colors[0], 'black')):
                    ax.text(x=0.97, y=0.99-0.03*ii, s=val, ma='right', ha='right', va='top', fontsize=self.fontsize, color=color, transform=ax.transAxes)
                    ii += 1
        
#        plt.tight_layout()
        self._save_fig(var+'_'+idate+'_grid', fig, pic_dpi=dpi, pic_subdir=pic_subdir)
        
        return
    
    
    def plot_profiles(self, var, idate, xval, zval, zdata, zterrain=np.empty(0), 
                      xyVar=np.empty(0), xyName='', grid_x=np.empty(0), grid_y=np.empty(0), draw_edge=False, profile_locs=[],
                      levs=[], xticklabels=[], cmap_rng=[], c_firstwhite=False, extend='neither', lw=None, ll_rng=[], dpi=900):
        
        if len(cmap_rng) > 0:
            cmap = self.icmap.set_cmap(rng=cmap_rng, c_firstwhite=c_firstwhite)
        else:
            cmap = copy.copy((cm.get_cmap("rainbow")))
            
#        sns.set_context('talk', font_scale=1.3)
        fig = plt.figure(figsize=(16, 16))
        if xyVar.size > 0:
            proj = ccrs.PlateCarree(central_longitude=0)
            ax1 = fig.add_axes([0.17, 0.34, 0.6, 0.6], projection=proj)
            
            data = np.ma.array(xyVar, mask=xyVar<(param.MeaningLess_val['missing']+1), fill_value=param.MeaningLess_val['missing'])
            tmplevs = param.desp_Vars[xyName][5]
            con = ax1.contourf(grid_x, grid_y, data, 
                          cmap=cmap,levels=tmplevs, norm=colors.BoundaryNorm(boundaries=np.array(tmplevs), ncolors=cmap.N), alpha=0.7, zorder=2, extend=extend)  # colors=param.collev1
            if draw_edge: self.draw_swath_edges(ax1, grid_x, grid_y, data, lw=lw)  
            
            self.__load_ax_settings__(fig, ax1, proj, con, xyName, rng=ll_rng)
            
        ax2 = fig.add_axes([0.17, 0.21, 0.6, 0.23])
        
        npts = xval.size
        xcoord = np.arange(npts)
        X, Y = np.meshgrid(xval, zval)
        con = ax2.contourf(X.T, Y.T, zdata, extend = extend, levels = levs,cmap=cmap)
        
        if zterrain.size > 0: ax2.fill_between(xcoord, zterrain, color='gray')
        ax2.set_xlim(0, npts-1)
    
        nd = npts//5      # 让tick有6个或7个
        ax2.set_xticks(xcoord[::nd])
        ax2.set_xticklabels(xticklabels)
        
        ax2.set_ylim(0, zval.max())
        ax2.set_yticks(zval[7::8])
#        ax2.set_yticklabels(np.arange(0,20,2))
        ax2.set_ylabel('Height [km]', fontsize='small')
    
        ax2.tick_params(labelsize=12)
        
        cb = fig.colorbar(con, shrink=1, pad=0.02, ax=ax2)
        cb.cmap.set_over('darkred')
        cb.ax.tick_params(direction='in', length=8)
        cb.ax.tick_params(which='minor', direction='in', length=5)
        cb.set_label(self.setlabel(param.desp_Vars[var]))
        
#        plt.tight_layout()
        
        self.draw_cross_line_on_map(ax1, profile_locs[0], profile_locs[1], proj)
        
        self._save_fig(var+'_'+idate+'profile', fig, pic_dpi=dpi)
        return 
    
    
    def draw_cross_line_on_map(self, ax, xp, yp, data_proj):
#    '''在GeoAxes的地图上画出所做剖面的水平线.'''
        
        xrng, yrng = ax.get_xlim(), ax.get_ylim()
        xs, ys = (xp-xrng[0]) / (xrng[1]-xrng[0]), (yp-yrng[0]) / (yrng[1]-yrng[0])
                
        ax.plot(xs[0], ys[0], 'r+', ms=2.5, transform=data_proj, alpha=0.7, zorder=3)
        ax.text(xs[0], ys[0], 'A', ha='right', va='bottom', fontsize=15, transform=ax.transAxes)
    
        ax.plot(xs[-1], ys[-1], 'r+', ms=2.5, transform=data_proj, alpha=0.7, zorder=3)
        ax.text(xs[-1], ys[-1], 'B', ha='left', va='top', fontsize=15, transform=ax.transAxes)
    
        ax.plot(xp, yp, c='grey',linestyle='--', lw=3, transform=data_proj, alpha=0.7, zorder=3)
        return
        
    
    def plot_scatter(self, var, idate, lats, lons, data, method, levs=[]):        
        fig, ax, proj = self.__load_panel_settings__()
        
        sca = ax.scatter(lons, lats, c=data, cmap='rainbow', norm=colors.Normalize(min(levs),max(levs),1), marker='.', edgecolors='none', s=0.1, alpha=1, zorder=2) # 

        self.__load_ax_settings__(fig, ax, proj, sca, var, kind=method.split('_')[-1])
        ax.set_title(method)
        self._save_fig(var+'_'+idate+'_scatters', fig)
        
        return
    
    
    def plot_hist2d(self, fn, data, xy=[], density=False, xyLOGtick=[1, True], xylabels=[], cbarlabel='', rngs=[], cmin=None,
                    coefs={}, ratio=100, intl=0.04):    
        sns.set_context('talk', font_scale=1.3)
        fig, ax = plt.subplots(figsize=(10, 8))
        cmaps = self.icmap.set_cmap(rng=[0,60], c_firstwhite=False)
        
        if len(rngs) == 0: 
            rngs = [data.min().min(), data.max().max()]
        bins = np.arange(np.floor(rngs[0]*ratio), np.ceil((rngs[1]+intl)*ratio), intl*ratio)/ratio
#        bins = np.power(10, (np.arange(np.floor(np.log10(rngs[0])*rato)/ratio, np.ceil(np.log10(rngs[1])*rato)/ratio+intl, intl)))
#        bins=100
        res = plt.hist2d(data[xy[0]].values, data[xy[1]].values, bins=bins, density=density, cmap=cmaps, norm=LogNorm(),
                         cmin=cmin)  # 
#        ax.set_xscale('log', base=10)
#        ax.set_yscale('log', base=10)
        ax.set_xlim(rngs[0], rngs[1]+intl)
        ax.set_ylim(rngs[0], rngs[1]+intl)
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xyLOGtick[0]))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(xyLOGtick[0]))       
        
        minorloc = []
        if xyLOGtick[-1]:
            for ii in range(np.floor(rngs[0]).astype(int), np.ceil(rngs[1]).astype(int)):
                minorloc.extend(list(range(2,10)*np.power(10, ii)))
            minorloc = np.log10(minorloc)
            minorloc = minorloc[np.logical_and(minorloc>=rngs[0], minorloc<=rngs[1])]
        
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minorloc))
            ax.yaxis.set_minor_locator(ticker.FixedLocator(minorloc))
        
#        ax.set_title('hist')
        cbar = plt.colorbar()
        if len(cbarlabel)>0: cbar.set_label(cbarlabel)   # 'Normalized frequency (%)'
            
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        
        ax.axline(xy1=(1, 1), xy2=(2,2), linestyle='--', color='grey')
        plt.tight_layout()
        
        if len(coefs) > 0:
            ha, va = 'right','bottom'
            [ax.text(x=0.98, y=0.02, s=key+f' {val :.4f}', fontsize=15, transform=ax.transAxes, ha=ha, va=va) for key, val in coefs.items()]
        
        self._save_fig(fn+'_hist2d', fig, pic_dpi=900)
        
        return
    
    
    def plot_PDF(self, fn, data, xy=[], density=False, xyLOGtick=[1, True], xylabels=[], rngs=[], ratio=100, intl=0.04):
        sns.set_context('talk', font_scale=1.2)
        fig, ax = plt.subplots(figsize=(10, 8))
        protion_top = 0.95
        
        if len(rngs) == 0: 
            rngs = [data[xy[0]].min(), data[xy[0]].max()]
        bins = np.arange(np.floor(rngs[0]*ratio), np.ceil((rngs[1]+intl)*ratio), intl*ratio)/ratio
#        bins = np.power(10, (np.arange(np.floor(np.log10(rngs[0])*rato)/ratio, np.ceil(np.log10(rngs[1])*rato)/ratio+intl, intl)))
#        bins=100
        res = sns.histplot(data, x=xy[0], hue=xy[1], bins=bins, element="poly", legend=False, log_scale=[False, True], ax=ax,
                           linewidth=0, alpha=0.7)
#        res.get_legend().set_title('')
        
        ax2 = plt.twinx()
        res_ = sns.ecdfplot(data, x=xy[0], hue=xy[1], legend=True, log_scale=[False, False], ax=ax2, alpha=0.7)   #  
        res_.get_legend().set_title('')
        
        ax.set_xlim(rngs[0], rngs[1]+intl)
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xyLOGtick[0]))
        
        minorloc = []
        if xyLOGtick[-1]:
            for ii in range(np.floor(rngs[0]).astype(int), np.ceil(rngs[1]).astype(int)):
                minorloc.extend(list(range(2,10)*np.power(10, ii)))
            minorloc = np.log10(minorloc)
            minorloc = minorloc[np.logical_and(minorloc>=rngs[0], minorloc<=rngs[1])]
        
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minorloc))
        else:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
            ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#            ax.yaxis.set_minor_locator(ticker.FixedLocator(minorloc))
        
#        ax.set_title('hist')
#        cbar = plt.colorbar()
#        if len(cbarlabel)>0: cbar.set_label(cbarlabel)   # 'Normalized frequency (%)'
            
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        
        plt.tight_layout()
        
        res_.hlines(0.95, xmin=res_.get_xlim()[0], xmax=res_.get_xlim()[1], linestyle='--', color='grey')
        line = res_.lines[0].get_xydata()        
        res_.vlines(line[np.where(line[:,1]>=protion_top)[0][0], 0], ymin=res_.get_ylim()[0], ymax=protion_top, linestyle='--', color=res_.lines[0].get_color())
        line = res_.lines[1].get_xydata()        
        res_.vlines(line[np.where(line[:,1]>=protion_top)[0][0], 0], ymin=res_.get_ylim()[0], ymax=protion_top, linestyle='--', color=res_.lines[1].get_color())
        
        self._save_fig(fn+'_PDF', fig, pic_dpi=900)
        
        return 
    
    
    def hist_stat(self, var, idate, xps, yps, bins=[], ranges=[]):
        if len(bins) == 0:
            bins = [round((np.max(xps)-np.min(xps))/param.grid_lon), round((np.max(yps)-np.min(yps))/param.grid_lat)]
        if len(ranges) == 0:
            ranges = [param.rng_lon, param.rng_lat]  # [[np.min(xps), np.max(xps)], [np.min(yps), np.max(yps)]]
        
        fig, ax, proj = self.__load_panel_settings__()
        SmpInGrid, xedges, yedges, img = plt.hist2d(xps.ravel(), yps.ravel(), bins=bins, range=ranges, cmap='rainbow',  cmin=1, alpha=0.7, zorder=2)  # , cmap='jet', normed=normed, norm=LogNorm()
        
        self.__load_ax_settings__(fig, ax, proj, img, var) # , rng=[param.rng_lon, param.rng_lat]
        ax.set_title('Sample counts')
        self._save_fig('Sample_counts_hist_'+idate, fig)
        return SmpInGrid.astype(np.int64)
    
    
    def plot_pf(self, var, idate, grid_x, grid_y, data, ellipses, orbit, levs=[], ll_rng=[]):
        if len(ellipses) == 0: return
        fig, ax, proj = self.__load_panel_settings__()
        datap = np.ma.array(data, mask=data<(param.MeaningLess_val['missing']+1), fill_value=param.MeaningLess_val['missing'])
        
        if len(levs) == 0:
            con = ax.contourf(grid_x, grid_y, datap, 
                          cmap='rainbow', alpha=0.7, zorder=2, extend='max') # norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=len(levs)-1), 
        else:
            con = ax.contourf(grid_x, grid_y, datap, 
                          cmap='rainbow',levels=levs, norm=colors.BoundaryNorm(boundaries=np.array(levs), ncolors=cm.rainbow.N), alpha=0.7, zorder=2)  # colors=param.collev1, extend='max'

        self.draw_ellipses(ax, ellipses)

        self.__load_ax_settings__(fig, ax, proj, con, var, rng=ll_rng)
        ax.set_title('orbit=%02i' % orbit)
        self._save_fig(var+'_'+idate+'_pf_%02i' % orbit, fig)
        
        return
    
    
    def draw_ellipses(self, ax, ellipses, lw=0.2, colors=[]):
        if lw==None: lw=0.2
        if len(colors) == 0: 
            colors = ['black']*len(ellipses)
        else:
            colors = colors*len(ellipses)
        for idx, vals in enumerate(ellipses):
            width, height = vals[1]
#            if width <= 0.01: width = 0.01
#            if height < 0.005: height = min(0.5*width, 0.01)
            
            ell1 = Ellipse(xy = vals[0], width = width, height = height, angle = vals[2],
                           color=colors[idx], fill=False, linewidth=lw, linestyle='-', alpha=1, zorder=3)
            ax.add_patch(ell1)
        return
    
    
    def draw_polygon(self, ax, polygons, lw=0.3, colors=[]):
        if lw==None: lw=0.2
        if len(colors) == 0: 
            colors = ['black']*len(polygons)
        else:
            colors = colors*len(polygons)
        
        for idx, val in enumerate(polygons):
            
            poly = Polygon(xy = np.array(val), 
                           color=colors[idx], fill=False, linewidth=lw, linestyle='-', alpha=1, zorder=3)
            ax.add_patch(poly)
        '''
        polys = [Polygon(xy= for val in polygons]
                 
        ax.add_collection(PatchCollection(polys, 
                                  facecolor='grey',
                                  edgecolor='red',
                                  linewidth=lw,
                                  alpha=1,
                                  zorder=3))
        '''
        return


    def draw_swath_edges(self, ax, grid_x, grid_y, data, lw=0.2):
        if lw==None: lw=0.3    
        if isinstance(data, np.ma.core.MaskedArray):
            tmpdata = data.filled()
        else:
            tmpdata = data
        ax.contour(grid_x, grid_y, tmpdata, levels=[param.MeaningLess_val['missing']], colors='k', linestyles='--', linewidths=lw, alpha=0.7, zorder=3)
        return 
    
    
    def _save_fig(self,fnn, fig, pic_dpi=1500, pic_subdir=''):  
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html#matplotlib.pyplot.savefig
#        plt.savefig(os.path.join(param.op_plot,'.'.join([fnn,'ps'])),bbox_inches='tight',transparent=False, pad_inches = 0 )  # too big   dpi=600,
        '''
        pdf = PdfPages(os.path.join(param.op_plot,'.'.join([fnn,'pdf'])))
        pdf.savefig(fig, bbox_inches='tight', transparent=False, pad_inches = 0)  # param.pic_dpi,
        pdf.close()
        '''
        DIR = os.path.join(param.op_plot,pic_subdir)
        if not os.path.exists(DIR): os.makedirs(DIR)
        fn = os.path.join(DIR,'.'.join([fnn,'png']))
        plt.savefig(fn, dpi=pic_dpi, bbox_inches='tight', transparent=False, pad_inches = 0)  # 
        
#        fn = os.path.join(param.op_plot,'.'.join([fnn,'svg']))
#        plt.savefig(fn, bbox_inches='tight', pad_inches = 0, transparent=False)  # dpi=pic_dpi, 

#        plt.show()
        plt.close()
        return  
    
    
    def setlabel(self,var,NoUnit=False):
        if len(var[1]) == 0 or NoUnit == True:
            item = var[0]
        else:
            if type(var) == tuple:
                item = var[0]+param.blank+var[1]
            else:   
                item = var[0][0]+param.blank+var[1][0]
        return item
    