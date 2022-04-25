# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:08:13 2022

@author: Gabriel
"""

from osgeo import gdal
import joblib
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

#Classificando
root = tk.Tk()    
entrada = filedialog.askopenfilename(filetypes=(("tif files","*.tif"),("All files","*.*")))
root.destroy()

modelo = joblib.load('D:/TCC/tcc_model_16bits_extra.pkl')
saida = 'D:/TCC/IMAGENS/Classificadas3/C_' + os.path.basename(entrada)

img = gdal.Open(entrada,gdal.GA_ReadOnly)

l,c = img.RasterYSize,img.RasterXSize

bandas = img.RasterCount

geo_transform = img.GetGeoTransform()
projection = img.GetProjectionRef()

array = img.ReadAsArray()

a = np.reshape(array, (bandas,l*c)).T

clf = modelo.predict(a)

img_clf = clf.reshape((l,c))

#Salvando
def createGeotiff(outRaster, data, geo_transform, projection):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Int32)
    rasterDS.SetProjection(projection)
    rasterDS.SetGeoTransform(geo_transform)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    rasterDS = None


#export classified image
createGeotiff(saida,img_clf,geo_transform,projection)
