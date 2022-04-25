# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:35:22 2022

@author: Gabriel
"""

from osgeo import gdal
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()    
img_new = filedialog.askopenfilename(filetypes=(("tif files","*.tif"),("All files","*.*")))
img_old = filedialog.askopenfilename(filetypes=(("tif files","*.tif"),("All files","*.*")))
root.destroy()

saida = 'D:/TCC/IMAGENS/Classificadas/Reprocessada/Change_' + os.path.basename(img_old)

img_new = gdal.Open(img_new,gdal.GA_ReadOnly)
img_old = gdal.Open(img_old,gdal.GA_ReadOnly)


l,c = img_new.RasterYSize,img_new.RasterXSize

array_new = img_new.ReadAsArray()

array_old = img_old.ReadAsArray()

geo_transform = img_new.GetGeoTransform()
projection = img_new.GetProjectionRef()

#re_cls = np.zeros([l,c])

for i in range(0,l):
    for j in range(0,c):
        if array_new[i,j] == 2 and array_old[i,j] != 2:
            array_old[i,j] = 2
        if array_new[i,j] == 3 and array_old[i,j] != 3:
            array_old[i,j] = 3

array_old = array_old.astype('uint8')
kernel = np.ones([3,3])
array_old = cv2.morphologyEx(array_old, cv2.MORPH_CLOSE, kernel)

driver = gdal.GetDriverByName('GTiff')
rows, cols = array_old.shape
rasterDS = driver.Create(saida, cols, rows, 1, gdal.GDT_Int32)
rasterDS.SetProjection(projection)
rasterDS.SetGeoTransform(geo_transform)
band = rasterDS.GetRasterBand(1)
band.WriteArray(array_old)
rasterDS = None