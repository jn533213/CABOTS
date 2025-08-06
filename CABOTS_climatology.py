import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata
import cmocean as cm
import shapely.geometry
import rasterio.features
import warnings
import time as tt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cartopy.crs as ccrs
import cartopy.feature as cfeature



'''
The purpose of this script is to create a climatology over a specified time period.
'''

#Define a climatology year range
years = np.arange(1991,2020+1).astype(str)

#Cycle through each of the months of interest
months = np.arange(1,12+1)
for month in months:

	#Import the files of interest
	path_input = '/home/jcoyne/Documents/Bottom_Stats/climatology/monthly/'
	ds = xr.open_mfdataset([path_input+i+'_'+"%.2d" % month+'.nc' for i in years])

	#Average along the time axis
	ds_temp = ds.bottom_temperature
	temp_clim = ds_temp.mean(axis=0).values
	ds_saln = ds.bottom_salinity
	saln_clim = ds_saln.mean(axis=0).values

	#Determine the average year for each grid cell
	temp_years = ds_temp.values
	saln_years = ds_saln.values
	for i in np.arange(years.size):
		temp_years[i][~np.isnan(temp_years[i])] = years[i]
		saln_years[i][~np.isnan(saln_years[i])] = years[i]
	warnings.filterwarnings("ignore", category=RuntimeWarning) 
	temp_years_mean = np.nanmean(temp_years, axis=0)
	temp_years_stdv = np.nanstd(temp_years, axis=0)
	saln_years_mean = np.nanmean(saln_years, axis=0)
	saln_years_stdv = np.nanstd(saln_years, axis=0)

	#Record both the average temperature, grid_year mean and stdv in a netcdf
	path_output = '/home/jcoyne/Documents/Bottom_Stats/climatology/'

	#Set up the .nc file
	nc_out = nc.Dataset(path_output+'bottomclim_'+"%.2d" % month+'.nc','w')

	#File information
	nc_out.title = 'Bottom Temperature and Salinity Climatology' #Temporary title for the .nc file
	nc_out.institution = 'Northwest Atlantic Fisheries Centre, Fisheries and Oceans Canada'
	nc_out.description = 'Output by jonathan.coyne@dfo-mpo.gc.ca'
	nc_out.history = 'Created ' + tt.ctime(tt.time())

	#Create dimensions
	x = nc_out.createDimension('x', temp_clim.shape[1])
	y = nc_out.createDimension('y', temp_clim.shape[0])

	#Create coordinate variables
	xs = nc_out.createVariable('x', np.int32, ('x',))
	ys = nc_out.createVariable('y', np.int32, ('y',))

	#Create 2D variables
	bottom_temp = nc_out.createVariable('mean_bottom_temperature', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	bottom_saln = nc_out.createVariable('mean_bottom_salinity', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	bottom_lons = nc_out.createVariable('longitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	bottom_lats = nc_out.createVariable('latitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	temp_years_means = nc_out.createVariable('temp_mean_year', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	temp_years_stdvs = nc_out.createVariable('temp_stdv_mean', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	saln_years_means = nc_out.createVariable('saln_mean_year', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	saln_years_stdvs = nc_out.createVariable('saln_stdv_mean', np.float32, ('y','x'), zlib=True, fill_value=-9999)

	#Variable Attributes
	bottom_lats.units = 'degree_north'
	bottom_lons.units = 'degree_east'

	#Fill in the 2D structure
	bottom_temp[:,:] = temp_clim[:,:]
	bottom_saln[:,:] = saln_clim[:,:]
	bottom_lons[:,:] = ds.longitude[0].values[:,:]
	bottom_lats[:,:] = ds.latitude[0].values[:,:]
	temp_years_means[:,:] = temp_years_mean[:,:]
	temp_years_stdvs[:,:] = temp_years_stdv[:,:]
	saln_years_means[:,:] = saln_years_mean[:,:]
	saln_years_stdvs[:,:] = saln_years_stdv[:,:]

	#Fill in the dimension variables
	xs[:] = np.arange(temp_clim.shape[1])
	ys[:] = np.arange(temp_clim.shape[0])

	#Save and close the .nc file
	nc_out.close()
	print(str(month)+' done!')

