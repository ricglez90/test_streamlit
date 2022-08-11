#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:17:13 2021

@author: cacie
"""

import streamlit as st
import numpy as np
import time
import cv2
from PIL import Image
#from color_segmentation import global_threshold_gs, k_means_segmentation, fuzzy_c_means_color_segmentation, larvae_counter
import random
import pandas as pd
#import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from aids import AgroImage
from segmentation import global_threshold_gs, k_means_segmentation, fc_means_color_segmentation

st.title("Color segmentation framework")
st.write("This framework is applied on agricultural images, grape bunches specifically, but you can apply it with another type of images.")

FRAME_WINDOW = st.image([])

filename = st.file_uploader("File")
opt = ""


if filename is not None:
    I = Image.open(filename).convert("RGB")
    b,g,r = I.split()
    I_np = np.array(Image.merge("RGB", (b,g,r)))
    A = AgroImage(I_np)
    left_column, right_column = st.columns(2)
    
    

    
    with left_column:
        st.subheader("Original image")
        st.image(I)
        
    config = st.sidebar.selectbox("Color configuration",["Color Index","Color Spaces"])
    if config == "Color Index":   
        with right_column:
            I_process = np.zeros_like(I_np)
            
            opt = st.sidebar.radio("Pick an index",A.names,key = [i for i in range(10)])
            A.color_index(opt)
                
            I_process = Image.fromarray(cv2.applyColorMap(A.CI,cv2.COLORMAP_JET))
            st.subheader(opt)
            st.image(I_process)
        if opt is not "":
            sub = A.concepts(opt)
            st.subheader(sub[0])
            st.write(r''+sub[1])
        th = st.sidebar.slider("Single threshold", min_value=0, max_value=255, value=128)
        I_seg = global_threshold_gs(A.CI,th)
        CI = I_np.copy()
        CI[I_seg==0] = 0
        I_proc = Image.fromarray(CI)
        st.subheader("Segmented image")
        st.image(I_proc)
                    
        
    elif config == "Color Spaces":
        with right_column:
            opt = st.sidebar.radio("Pick a space",A.spaces)
            if opt == "HSV":
                C1,C2,Int = A.color_sapces(opt)
            elif opt == "LAB":                
                Int,C1,C2 = A.color_sapces(opt)
                
            segmenter = st.sidebar.selectbox("Select algorithm",["K-Means","FC-Means"])
            fl = st.sidebar.checkbox("Only color features")
            k = st.sidebar.slider("Clusters",min_value=1,max_value=100,value=5)
            iter = st.sidebar.slider("Iterations",min_value=1,max_value=100,value=10)
            C_feat = np.dstack((C1,C2)) if fl else np.dstack((C1,C2,Int))
            
            if segmenter == "K-Means":
                I_seg = k_means_segmentation(C_feat,k,iter)
            elif segmenter == "FC-Means":
                p = st.sidebar.slider("P",min_value=1,max_value=10,value=2)
                I_seg = fc_means_color_segmentation(C_feat,k,p,iter)
            
            I_proc = Image.fromarray(cv2.applyColorMap(I_seg.astype(np.uint8),cv2.COLORMAP_JET))
            st.subheader("Segmented image")
            st.image(I_proc)
                    
                
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Color feature 1")
            st.image(Image.fromarray(cv2.applyColorMap(C1,cv2.COLORMAP_JET)))
        with col2:
            st.subheader("Color feature 2")
            st.image(Image.fromarray(cv2.applyColorMap(C2,cv2.COLORMAP_JET)))
        with col3:
            st.subheader("Intensity")
            st.image(Image.fromarray(Int))
            
        
            
    
        
  
            

##                    I_process = Image.fromarray(A.CI[0][0])
#        if len(A.CI) == 1:
#            th = st.sidebar.slider("Threshold", min_value=0, max_value=255, value=128)
#            st.subheader("Color Index sample")
#            st.image(I_process)
#            
#            
#            I_process = Image.fromarray(global_threshold_gs(A.CI[0][0],th))
#            st.subheader("Segmented")
#            st.image(I_process)
#        elif len(A.CI) >= 2:
#            st.sidebar.slider("Threshold", min_value=0, max_value=255, value=128)
#                st.write(len(A.CI))
#        st.write(color_indx_cbox)
#        if side_options == "Thresholding":
#        th =  st.sidebar.slider("Threshold", min_value=0, max_value=255, value = int(0.6*255))
#        st.subheader("Single threshold")
#        I_seg = global_threshold_gs(I_np,th)
#            random_hex = []
#            clusters = 2
#            for i in range(2):
#                col = st.sidebar.color_picker("Cluster "+str(i+1))
#                random_hex.append(tuple(int(col.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)))
#                I_process[I_seg == i] = tuple(int(col.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
#        I_process = Image.fromarray(I_seg)
                        
#        elif side_options == "K-means":
#            c_space = st.sidebar.selectbox("Color space",("HSV","LAB"))
#            chroma = st.sidebar.radio("Only chroma",(True,False))
#            iterations = st.sidebar.slider("Iterations", min_value=0, max_value=100, value = 10)
#            clusters = st.sidebar.slider("Clusters", min_value=0, max_value=20, value = 5)
#            st.subheader("K-means segmentation")
#            
#            mu, I_seg = k_means_segmentation(I_np, k = clusters, iterar=iterations, color_space=c_space, chroma=chroma)
#            
#           
                
#            colors = np.random.randint(0,255,size=(clusters,3))
            
#            random_hex = []
#            for i in range(clusters):
#                col = st.sidebar.color_picker("Cluster "+str(i+1))
#                random_hex.append(tuple(int(col.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)))
#                I_process[I_seg == i] = tuple(int(col.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
#                
#            I_process = Image.fromarray(np.array(I_process,dtype=np.uint8))
                        
#        elif side_options == "Fuzzy-C-means":
#            c_space = st.sidebar.selectbox("Color space",("HSV","LAB"))
#            chroma = st.sidebar.radio("Only chroma",(True,False))
#            iterations = st.sidebar.slider("Iterations", min_value=0, max_value=100, value = 10)
#            clusters = st.sidebar.slider("Clusters", min_value=0, max_value=20, value = 5)
#            p_val = st.sidebar.slider("P value", min_value=0, max_value=10, value = 2)
#            st.subheader("FC-means segmentation")
            
#            mu, I_seg = fuzzy_c_means_color_segmentation(I_np, k = clusters, iters=iterations, p=p_val, color_space=c_space, chroma=chroma)
            
#            colors = np.random.randint(0,255,size=(clusters,3))
#            
#            for c in range(clusters):
#                I_process[I_seg == c] = colors[c]
#                
#            I_process = Image.fromarray(np.array(I_process,dtype=np.uint8))
            
##        random_hex = []
#        for i in range(clusters):
##            col = st.sidebar.color_picker("Cluster "+str(i+1))
##            random_number = random.randint(0,16777215)
##            hex_number = str(hex(random_number))
#            hex_number = random_hex[i]
##            hex_number ='#'+ hex_number[2:]
##            random_hex.append(hex_number)
#            random_hex.append(tuple(int(hex_number.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)))
#            I_process[I_seg == i] = tuple(int(hex_number.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
#        st.image(I_process)
        
   
#    st.subheader("Larvae detection")
##        st.write(len(random_hex))
#    if I_process is not None:
#        delta = st.sidebar.slider("Delta", 50, 500, value = 120)
#        neig = st.sidebar.slider("Neighboorhood", 1,10, value = 3)
#        min_dist = st.sidebar.slider("Min. distance", 1,250, value = 40)
##        st.write(I_np.shape, th/255.0)
#        larvas , lar2,ind,pts = larvae_counter(I_np, threshold= th/255.0, delta=delta, neighbors= neig, min_dist=min_dist)
#        st.image(Image.fromarray(larvas))
#        st.write(ind[ind[:,0]==0])
#        st.write(pts[0])
#        for nm in lar2:
#            if nm[1] == 1:
#                st.write(nm[0])
#                break
#        cluster_ids = np.unique(np.array(I_seg))
#        st.write(cluster_ids)
#        bars = []
#        colorss = []
##        for i in cluster_ids:
##            bars.append(I_seg[I_seg==i].shape[0] )#/ float (I_seg.shape[0]*I_seg.shape[1])
##            st.write("rgb"+str(random_rbg[i]))
##            colorss.append("rgb"+str(random_rbg[i]))
#        
##        group_labels = ["Cluster "+str(i+1) for i in cluster_ids]
#        
#        fig = go.Pie(labels = group_labels, values = bars, marker_colors=colorss)
#        fig = go.Figure(fig)
#        fig.update(layout_showlegend=False)
#        st.plotly_chart(fig)#,use_container_width=True
#
#
