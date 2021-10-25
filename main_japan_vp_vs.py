# -*- coding: utf-8 -*-
# Written by Jisong Kim (jisong@unist.ac.kr)

import pickle
import PySimpleGUI as sg
from pickle import load
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

######################
##### Load model #####
######################
models = pickle.load(open(r"models.dat", "rb"))
encoders = pickle.load(open(r"encoders.dat", "rb"))

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')
    
def draw(X, Y):
    new_values, new_depthes = [], []
    for num, (value, depth) in enumerate(zip(X, Y)):
        if num == 0:
            new_values.append(value)
            new_depthes.append(0)
            new_values.append(value)
            new_depthes.append(depth)
        else:
            if new_values[-1] != value:
                new_values.append(value)
                new_depthes.append(new_depthes[-1])
            new_values.append(value)
            new_depthes.append(depth)
    return new_values, new_depthes    

def draw_config(ax):
    ax.minorticks_on()

    for fr in ["top", "bottom", "left", "right"]:
        ax.spines[fr].set_color("black")
        ax.spines[fr].set_linewidth(0.5)

    ax.xaxis.set_tick_params(which='both', bottom=True)
    ax.yaxis.set_tick_params(which='both', bottom=True)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis = 'both', labelsize=12, which = 'major', length=5, width=0.5) 
    ax.tick_params(axis = 'both', labelsize=12, which = 'minor', length=3, width=0.5) 

    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)', labelpad=7, fontsize=12, font='Arial')
    ax.set_xlabel(r'Velocity (km/s)', fontsize=12, font='Arial')
    ax.legend()  
    return ax

sg.theme('Dark Blue 3')

part_profile = [[sg.Frame(layout=
                  [
                      [sg.Text('Software developed by Jisong Kim (jisong@unist.ac.kr)')],
                      [sg.Text('Ulsan National Institute of Science and Technology (UNIST)')],
                      [sg.Text('Ulsan, South Korea, 44919')]
                  ], title='Profile', title_color='white')]]
    
part_open_file_knet = [[sg.Text('Open data file:'), sg.In(size=(30,), key='-FA1-'), 
                        sg.FileBrowse(), sg.Button('Open', key='-FA2-')],
                       [sg.Text('Data processing result:'), sg.In(size=(20,), key='-FA3-')],
                       [sg.Button('Predict', key='-FA4-'), sg.Button('Clear all', key='-FA5-')],
                       [sg.Table(values=[[None, None]], 
                           headings=['Depth (m)', 'Vp (m/s)', 'Vs (m/s)'], 
                           num_rows=25, 
                           auto_size_columns=False, 
                           key='-FA6-'),
                       sg.Frame(layout=[[sg.Canvas(size=(400, 400), key='-FA7-')]],
                                title='Figure', title_color='white')]]

part_open_file_kiknet = [[sg.Text('Open data file:'), sg.In(size=(30,), key='-FB1-'), 
                          sg.FileBrowse(), sg.Button('Open', key='-FB2-')],
                         [sg.Text('Data processing result:'), sg.In(size=(20,), key='-FB3-')],
                         [sg.Button('Predict', key='-FB4-'), sg.Button('Clear all', key='-FB5-')],
                         [sg.Table(values=[[None, None]], 
                           headings=['Depth (m)', 'Vp (m/s)', 'Vs (m/s)'], 
                           num_rows=25, 
                           auto_size_columns=False, 
                           key='-FB6-'),
                          sg.Frame(layout=[[sg.Canvas(size=(400, 400), key='-FB7-')]],
                                   title='Figure', title_color='white')]]

tabgrp = [[sg.TabGroup([[sg.Tab('K-net', part_open_file_knet, border_width =10, ),
                        sg.Tab('KiK-net', part_open_file_kiknet, border_width =10, ),
                        sg.Tab('Author Profile', part_profile)]],
                       )]] 

window = sg.Window('Predict Vp, Vs in Japan', tabgrp)

while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED:
        break
    
    # K-net
    elif event == '-FA2-':
        df = pd.read_excel(values.get('-FA1-'))
        df_sr = encoders['soil_rock_type_knet'].transform(df['soil_rock_type'])
        df_geo = encoders['geology_knet'].transform(df['geology'])
        idx = ['latitude', 'longitude', 'slope', 'elevation', 'geology_1',
               'geology_2', 'geology_3', 'geology_4', 'geology_5',
               'geology_6', 'geology_7', 'soil_rock_type_1', 'soil_rock_type_2',
               'soil_rock_type_3', 'soil_rock_type_4', 'n_value', 'density', 'depth']
        df = pd.concat([df, df_sr, df_geo], axis=1)[idx]
        window['-FA3-'].update('Success')
        
    elif event == '-FA4-':
        res_vp = models['knet_vp'].predict(df)
        res_vs = models['knet_vs'].predict(df)
        
        window['-FA6-'].update(values=list(zip(df['depth'], res_vp, res_vs)))
        
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        X, Y = draw(res_vp, df['depth'])
        ax.plot(X, Y, color='red', label = r'$V_{P}^{est}$', linestyle='-', linewidth=1.0)
        X, Y = draw(res_vs, df['depth'])
        ax.plot(X, Y, color='blue', label = r'$V_{S}^{est}$', linestyle='-', linewidth=1.0)
        
        ax = draw_config(ax)    

        ax.set_xlim(-500, 4500)
        ax.set_xticks(np.arange(0, 4001, 1000))
        tick_labels = ax.get_xticks().tolist()
        tick_labels = [int(i/1000) for i in tick_labels]
        ax.set_xticklabels(tick_labels)
        
        plt.grid()
        plt.tight_layout()
        fig_canvas_agg = draw_figure(window['-FA7-'].TKCanvas, fig)
        
    elif event == '-FA5-':
        window['-FA6-'].update('')
        window['-FA3-'].update('')
        delete_figure_agg(fig_canvas_agg)
        
    # KiK-net
    elif event == '-FB2-':
        df = pd.read_excel(values.get('-FB1-'))
        df_sr = encoders['soil_rock_type_kiknet'].transform(df['soil_rock_type'])
        df_geo = encoders['geology_kiknet'].transform(df['geology'])
        idx = ['latitude', 'longitude', 'slope', 'elevation', 'geology_1',
               'geology_2', 'geology_3', 'geology_4', 'geology_5',
               'geology_6', 'geology_7', 'soil_rock_type_1', 'soil_rock_type_2', 
               'soil_rock_type_3', 'soil_rock_type_4','soil_rock_type_5', 
               'soil_rock_type_6', 'soil_rock_type_7', 'soil_rock_type_8', 
               'soil_rock_type_9', 'soil_rock_type_10', 'depth']
        df = pd.concat([df, df_sr, df_geo], axis=1)[idx]
        window['-FB3-'].update('Success')
        
    elif event == '-FB4-':
        res_vp = models['kiknet_vp'].predict(df)
        res_vs = models['kiknet_vs'].predict(df)
        
        window['-FB6-'].update(values=list(zip(df['depth'], res_vp, res_vs)))
        
        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        X, Y = draw(res_vp, df['depth'])
        ax.plot(X, Y, color='red', label = r'$V_{P}^{est}$', linestyle='-', linewidth=1.0)
        X, Y = draw(res_vs, df['depth'])
        ax.plot(X, Y, color='blue', label = r'$V_{S}^{est}$', linestyle='-', linewidth=1.0)
        
        ax = draw_config(ax)    

        ax.set_xlim(-500, 5500)
        ax.set_xticks(np.arange(0, 5001, 1000))
        tick_labels = ax.get_xticks().tolist()
        tick_labels = [int(i/1000) for i in tick_labels]
        ax.set_xticklabels(tick_labels)
        
        plt.grid()
        plt.tight_layout()
        fig_canvas_agg = draw_figure(window['-FB7-'].TKCanvas, fig)
        
    elif event == '-FB5-':
        window['-FB6-'].update('')
        window['-FB3-'].update('')
        delete_figure_agg(fig_canvas_agg)