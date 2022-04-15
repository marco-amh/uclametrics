# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:45:57 2022

@author: marco
"""

import numpy as np
import pandas as pd
from bubbly.bubbly import bubbleplot
import seaborn as sns
import matplotlib.font_manager as font_manager
from plotly.offline import iplot
import matplotlib
import matplotlib.pyplot as plt

# Functions

def banxico_plot(df, y, hue, title,pal, ylim_up=10, ylim_lo=0, xlim_lo=0, xlim_up=100,xstep=10,ystep=10,leg_col=1, fig_hei=12.2, fig_wid=14.9):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style('ticks')
    sns.lineplot(x ='Fecha', y=y, hue=hue, data=df, linewidth=1.5,palette=pal)
    sns.despine()
    leg = plt.legend(frameon=False, title='', loc='best',ncol=leg_col,prop=font) #edgecolor='white'
    for i in leg.legendHandles:
        i.set_linewidth(1)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel('', fontsize=0, **banxicofont)
    plt.ylabel('', fontsize=0, **banxicofont)
    plt.xticks(np.arange (xlim_lo,xlim_up+1,step = xstep), rotation=90)
    plt.yticks(np.arange (ylim_lo,ylim_up+1,step = ystep))
    plt.ylim([ylim_lo,  ylim_up])
    plt.xlim([xlim_lo,  xlim_up])
    #fig.savefig('test.jpg')

def banxico_uni(df, y, title,pal, ylim_up=10, ylim_lo=0, xlim_lo=0, xlim_up=100,xstep=10,ystep=10,leg_col=1, fig_hei=12.2, fig_wid=14.9):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style('ticks')
    sns.lineplot(x = df.index, y=y, data=df, linewidth=1.5,palette=pal)
    sns.despine()
    leg = plt.legend(frameon=False, title='', loc='best',ncol=leg_col,prop=font) #edgecolor='white'
    for i in leg.legendHandles:
        i.set_linewidth(1)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel('', fontsize=0, **banxicofont)
    plt.ylabel('', fontsize=0, **banxicofont)
    plt.xticks(np.arange (xlim_lo,xlim_up+1,step = xstep), rotation=90)
    plt.yticks(np.arange (ylim_lo,ylim_up+1,step = ystep))
    plt.ylim([ylim_lo,  ylim_up])
    plt.xlim([xlim_lo,  xlim_up])

def banxico_boxplot(df,y,x, title, pal, ylim_up = 10, ylim_lo = 0, ystep = 10,leg_col=1, fig_hei=12.2, fig_wid=14.9):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style('ticks')
    sns.boxplot( y=y, x=x, data=df, linewidth=1,palette=pal)
    #order=["Dinner", "Lunch"],orient='h'
    sns.despine()
    leg = plt.legend(frameon=False, title='', loc='best',ncol=leg_col, prop=font) #edgecolor='white'
    for i in leg.legendHandles:
        i.set_linewidth(1)
    plt.suptitle(title, fontsize=10)
    plt.xlabel('', fontsize=0, **banxicofont)
    plt.ylabel('', fontsize=0, **banxicofont)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(ylim_lo,ylim_up+1, step=ystep))
    plt.ylim([ ylim_lo,  ylim_up])
    #fig.savefig('test.jpg')

def banxico_bubble(df,y,x,hue, title,  pal, ylim_lo = 0, ylim_up = 10, xlim_lo = 0, xlim_up = 100, xstep = 10,ystep = 10,leg_col=1, fig_hei=12.2, fig_wid=14.9):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style("ticks")
    #sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue=hue, legend='auto', sizes=(20, 2000),palette=pal)
    sns.set_context(font_scale=1.5, rc={"lines.linewidth": 10.5})
    sns.despine()
    plt.legend( edgecolor='white',title='', loc='best',ncol=leg_col,prop=font)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel('Inversión / PIB', fontsize=10, **banxicofont)
    plt.ylabel('Emisión interna / Financiamiento total', fontsize=10, **banxicofont)
    plt.xticks(rotation=90)
    plt.yticks(np.arange(ylim_lo, ylim_up+1, step = ystep))
    plt.ylim([ylim_lo, ylim_up])
    plt.show()

def banxico_regplot(df,y,x,hue, title, pal, xtit='X', ytit='Y',  ylim_lo = 0, ylim_up = 10, xlim_lo = 0, xlim_up = 100, xstep = 10,ystep = 10,leg_col=1,rot=90, fig_hei=11.8, fig_wid=10.27):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style('ticks')#sns.set_style("whitegrid")
    #sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    sns.lmplot(data = df, x = x, y = y, hue = hue, legend=False, ci=0,scatter_kws={"s": 7}, palette = pal)
    sns.despine()
    plt.legend( edgecolor = 'white', title = '', loc = 'best', ncol = leg_col, prop = font)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel(xtit, fontsize=10, **banxicofont)
    plt.ylabel(ytit, fontsize=10, **banxicofont)
    plt.xticks(np.arange(xlim_lo, xlim_up+1, step = xstep), rotation = rot)
    plt.yticks(np.arange(ylim_lo, ylim_up+1, step = ystep))
    plt.ylim([ylim_lo, ylim_up])
    plt.xlim([xlim_lo, xlim_up])
    plt.show()

def banxico_bar(df,y1,y2,x,n,bot1, title, y1tit, y2tit, val1,val2,val3 ,ylim_lo = 0, ylim_up = 10, ystep = 10, leg_col=1,rot=90, fig_hei=11.8, fig_wid=10.27):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style("ticks")
    plt.bar(x = x, height = y1, data = df, color = 'green',  label = y1tit)
    plt.bar(x = x, height = y2, data = df, color = 'red', label = y2tit, bottom=bot1)
    for i in range(0,n):
        plt.text(i, val1[i],"{0:.0f}".format(df[y1].iloc[i:i+1].values[0]), horizontalalignment='center', verticalalignment='center', rotation=0, color='white',fontsize=9,fontweight='bold')
        plt.text(i, val2[i],"{0:.0f}".format(df[y2].iloc[i:i+1].values[0]), horizontalalignment='center', verticalalignment='center', rotation=0, color='white',fontsize=9,fontweight='bold')
        plt.text(i, val3[i],"{0:.0f}".format(df[y1].iloc[i:i+1].values[0] + df[y2].iloc[i:i+1].values[0]), horizontalalignment='center', verticalalignment='center', rotation=0, color='black',fontsize=9,fontweight='bold')
    sns.despine()
    leg = plt.legend(frameon=False, title='', loc='best',prop=font, ncol=leg_col) #edgecolor='white'
    for i in leg.legendHandles:
        i.set_linewidth(1)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel('', fontsize=10, **banxicofont)
    plt.ylabel('', fontsize=10, **banxicofont)
    plt.yticks(np.arange(ylim_lo, ylim_up+1, step = ystep))
    plt.xticks(rotation = rot)
    plt.ylim([ylim_lo, ylim_up])
    plt.show()

def banxico_bar_100(df,y1,y2,x,n,bot1, title, y1tit, y2tit, val1,val2,ylim_lo = 0, ylim_up = 10, ystep = 10, leg_col=1,rot=90, fig_hei=11.8, fig_wid=10.27):
    banxicofont = {'fontname':'Calibri'}
    font = font_manager.FontProperties(family='Calibri',style='normal', size=10) #weight='bold'
    plt.figure(figsize=(fig_hei/2.54, fig_wid/2.54))
    sns.set_style("ticks")
    plt.bar(x = x, height = y1, data = df, color = 'green',  label = y1tit)
    plt.bar(x = x, height = y2, data = df, color = 'red', label = y2tit, bottom=bot1)
    for i in range(0,n):
        plt.text(i, val1[i],"{0:.0f}".format(df[y1].iloc[i:i+1].values[0]), horizontalalignment='center', verticalalignment='center', rotation=0, color='white',fontsize=9,fontweight='bold')
        plt.text(i, val2[i],"{0:.0f}".format(df[y2].iloc[i:i+1].values[0]), horizontalalignment='center', verticalalignment='center', rotation=0, color='white',fontsize=9,fontweight='bold')
    sns.despine()
    leg = plt.legend(frameon=False, title='', loc='best',prop=font, ncol=leg_col) #edgecolor='white'
    for i in leg.legendHandles:
        i.set_linewidth(1)
    plt.suptitle(title, fontsize=10, **banxicofont)
    plt.xlabel('', fontsize=10, **banxicofont)
    plt.ylabel('', fontsize=10, **banxicofont)
    plt.yticks(np.arange(ylim_lo, ylim_up+1, step = ystep))
    plt.xticks(rotation = rot)
    plt.ylim([ylim_lo, ylim_up])
    plt.show()

 

#pal1 = ['red','navy','gold','green','purple','gray','orange','brown','pink']