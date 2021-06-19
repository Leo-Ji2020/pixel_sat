# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:22:20 2020

@author: aa
"""

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


class self_def_colormaps():
    def __init__(self):
        # https://www.sioe.cn/yingyong/yanse-rgb-16/
        self.cbx = colors.ColorConverter().to_rgb
        self.base_color_dict = {0: [self.cbx('white')],
                                1: [self.cbx('Gainsboro'), self.cbx('dimgray')],
                                2: [self.cbx('palegreen'), self.cbx('lime')],
                                3: [self.cbx('limegreen'), self.cbx('green')],
                                4: [self.cbx('lightcyan'), self.cbx('mediumblue')],
                                5: [self.cbx('yellow'), self.cbx('chocolate')],
                                6: [self.cbx('red'), self.cbx('darkred')],
                                7: [self.cbx('magenta'), self.cbx('purple')],
                                8: [self.cbx('indigo')]}
        self.self_def_intl = {1: 1.2, 
                              2: 0.4, 
                              3: 0.3, 
                              4: 2.3, 
                              5: 1.5, 
                              6: 1.5,
                              7: 1}
        self.head_tail = 0.001
        return
    
    
    def set_cmap(self, kind='pr', rng=[], ncol=-1, c_firstwhite=False):
        if ncol == -1: ncol=66
        if kind in ['pr']:
            rgb = self.rgb_pr(rng=rng, c_firstwhite=c_firstwhite)
            icmap=colors.ListedColormap(rgb,name='my_color')
        elif kind in ['pr1']: 
            icmap = self.rgb_pr2(rng=rng, ncol=ncol)
        elif kind in ['cs']:
            icmap = self.rgb_pr3(ncol=2)
        elif kind in ['cs0']:
            rgb = self.rgb_range()
            icmap = colors.LinearSegmentedColormap.from_list('my_color', rgb)
        else:
            icmap = plt.get_cmap('jet')
        return icmap
    
    
    def rgb_pr(self, rng=[], c_firstwhite=False):
        r = np.array([255,247, 240, 232, 225, 217, 210, 203, 195, 188, 180, 173, 165, 158, 151,
          133, 126, 119, 112, 105, 207, 181, 156, 131, 106,  81,  73,  65,  57,  50,   0,   0,   0,   0, 
            0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 205, 193, 182, 
          170, 159,  94,  91,  89,  86,  84, 255, 246, 238, 229, 221, 213])
   
        g = np.array([255,247, 240, 232, 225, 217, 210, 202, 195, 187, 180, 172, 165, 157, 150, 
          231, 221, 211, 201, 191, 216, 208, 200, 192, 184, 177, 145, 113,  81,  50,   0,   0,   1,   1, 
            2, 255, 243, 231, 220, 208, 147, 138, 130, 122, 114,  64,  53,  43,  32,  22,   0,   2,   5,  
            8,  11,  15,  18,  21,  24,  27, 133, 131, 130, 129, 128, 127])
   
        b = np.array([255,247, 239, 232, 224, 216, 209, 201, 193, 186, 178, 170, 163, 155, 148, 
          137, 128, 120, 112, 104, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 241, 227, 213, 
          199,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   5,  10,  15,  20,  25, 137, 145, 154, 162, 171, 180])
        
        if len(rng) == 0:
            res = np.vstack((r,g,b)).T
        else:
            if len(rng) == 1: 
                res = np.vstack((np.array([0,255]), np.array([0,0]), np.array([255,0]))).T
            else:
                if len(rng) == 2: rng = rng+[1]
                res = np.vstack((r[rng[0]:rng[1]][::rng[2]],g[rng[0]:rng[1]][::rng[2]],b[rng[0]:rng[1]][::rng[2]])).T
            if c_firstwhite:
                res = np.vstack((np.array([255,255,255])[np.newaxis,:],res))
        return res/255.
    
    
    def rgb_pr2(self, rng=[], ncol=-1):
        #cbx('white'),0.01,    
        seq = {}
        if len(rng) == 0:
            seq = self.calc_color_seqs(c_keys=list(np.arange(len(self.base_color_dict)-1)+1) )           
            res = self.make_colormap(seq, ncol=ncol)
        else:
            if rng[0] == -1: 
                res = np.vstack((np.array([0,255]), np.array([0,0]), np.array([255,0]))).T/255.
            else:
                color_dict = {}
                for ii in rng:
                    color_dict.update({ii: self.base_color_dict[ii]})
                                    
                seq = self.calc_color_seqsX(c_keys=rng)            
                res = self.make_colormap(seq, ncol=ncol)
        return res
    
    
    def rgb_pr3(self, ncol=2):
        mycolor = [self.cbx('lightgreen'),0.5, self.cbx('orange')]
        res = self.make_colormap(mycolor, ncol=ncol)
        return res
    
    
    def calc_color_seqs(self, c_keys):
        res = []
        maxrng, hts, first_white = 1., 0, 1
        if min(self.base_color_dict.keys()) in c_keys: 
            maxrng -= self.head_tail
            hts += 1
            first_white = 0
        if max(self.base_color_dict.keys()) in c_keys: 
            maxrng -= self.head_tail
            hts += 1
        
        intl = maxrng / (len(c_keys)-hts)
        
        for idx, key in enumerate(c_keys):
            if key == 0:
                res += self.base_color_dict[key] + [self.head_tail]
            else:
                res += self.base_color_dict[key] + [min(self.head_tail*(not first_white)+intl*(idx+first_white), 1.)]
        
        res = res[:-1]
        
        return res
    
    
    def calc_color_seqsX(self, c_keys):
        res = []
        maxrng, hts, first_white = 1., 0, 1
        if min(self.base_color_dict.keys()) in c_keys: 
            maxrng -= self.head_tail
            hts += 1
            first_white = 0
        if max(self.base_color_dict.keys()) in c_keys: 
            maxrng -= self.head_tail
            hts += 1
            
        c_cumsum = [0]
        for key in c_keys:
            c_cumsum.extend([c_cumsum[-1] + self.self_def_intl[key]])
            
        intl = {key: maxrng*c_cumsum[idx+1]/c_cumsum[-1] for idx, key in enumerate(c_keys)}
        
        for idx, key in enumerate(c_keys):
            if key == 0:
                res += self.base_color_dict[key] + [self.head_tail]
            else:
                res += self.base_color_dict[key] + [min(self.head_tail*(not first_white)+intl[key], 1.)]
        
        res = res[:-1]
        
        return res
    
    
    def make_colormap(self, seq, ncol=66):
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return colors.LinearSegmentedColormap('CustomMap', cdict, N=ncol)
    
    
    def rgb_range0(self):
        mycolor=['white',
                 'whitesmoke','grey',
                 'palegreen','green','darkgreen',
                 'aliceblue','deepskyblue','blue','darkblue',
                 'yellow','gold','darkorange',
                 'red','brown','darkred',
                 'pink','violet']
        return mycolor
    
    
    