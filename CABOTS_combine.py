import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata
import cmocean as cm
import shapely.geometry
import rasterio.features
import warnings
import time as tt
import pandas as pd
import skfuzzy as fuzz
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cartopy.crs as ccrs
import cartopy.feature as cfeature



'''
The purpose of this script is to combine bottom temperature and salinity by season.
This will be the final formatted data product. 
Ensure that meta-data is properly organized HERE.
'''

#Cycle through each season
#spring,summer,fall
season = 'spring'
if season=='spring':
	months_covered='4,5,6'
elif season=='summer':
	months_covered='7,8,9'
elif season=='fall':
	months_covered='10,11,12'

#Define the temperature and salinity paths
path = '/home/jcoyne/Documents/Bottom_Stats/adjusted/'+season+'/'
finl_path = '/home/jcoyne/Documents/Bottom_Stats/final_product/'

#Import the data of interest
ds = xr.open_mfdataset(path+'*.nc')

#Limit the spatial extent of SFT and SFS
lonLims = [-70,-42]
latLims = [39,80]
longitude = ds.longitude[0].values
latitude = ds.latitude[0].values
saln = ds.bottom_salinity.values
saln_adj = ds.bottom_salinity_adjusted.values
temp = ds.bottom_temperature.values
temp_adj = ds.bottom_temperature_adjusted.values

#Create filter
x_filter_min = np.where(longitude[0,:] >= lonLims[0])[0][0]
x_filter_max = np.where(longitude[0,:] <= lonLims[1])[0][-1]
y_filter_min = np.where(latitude[:,0] >= latLims[0])[0][0]
y_filter_max = np.where(latitude[:,0] <= latLims[1])[0][-1]
longitude = longitude[y_filter_min:y_filter_max,x_filter_min:x_filter_max]
latitude = latitude[y_filter_min:y_filter_max,x_filter_min:x_filter_max]
saln = saln[:,y_filter_min:y_filter_max,x_filter_min:x_filter_max]
saln_adj = saln_adj[:,y_filter_min:y_filter_max,x_filter_min:x_filter_max]
temp = temp[:,y_filter_min:y_filter_max,x_filter_min:x_filter_max]
temp_adj = temp_adj[:,y_filter_min:y_filter_max,x_filter_min:x_filter_max]


#Create the combined dataset
nc_out = nc.Dataset(finl_path+'CABOTS_'+season+'.nc','w')

#File information
nc_out.title = 'CABOTS Bottom Stats - '+season #Temporary title for the .nc file
nc_out.institution = 'Northwest Atlantic Fisheries Centre, Fisheries and Oceans Canada'
nc_out.description = 'CABOTS (Canadian Atlantic Shelf Bottom Ocean Temperature & Salinity) seasonal ('+season+') output.'
nc_out.season = season+': months covered '+months_covered
nc_out.source = 'CABOTS'
nc_out.reference = 'Coyne, J., Cyr, F. (2024). Canadian Atlantic Bottom Temperature and Salinity. Federated Research Data Repository. DOI: 10.20383/103.0969'
nc_out.output = 'Output by jonathan.coyne@dfo-mpo.gc.ca'
nc_out.history = 'Created ' + tt.ctime(tt.time())

#Create dimensions
time = nc_out.createDimension('TIME', None) #use date2 for this
x = nc_out.createDimension('X', longitude.shape[1])
y = nc_out.createDimension('Y', latitude.shape[0])
times = nc_out.createVariable('TIME', np.float64, ('TIME',))
xs = nc_out.createVariable('X', np.int32, ('X',))
ys = nc_out.createVariable('Y', np.int32, ('Y',))

#Create 2D variables
bottom_saln = nc_out.createVariable('BOTTOM_SALINITY', np.float32, ('TIME','Y','X'), zlib=True, fill_value=-9999)
bottom_saln_adjusted = nc_out.createVariable('BOTTOM_SALINITY_ADJUSTED', np.float32, ('TIME','Y','X'), zlib=True, fill_value=-9999)
bottom_temp = nc_out.createVariable('BOTTOM_TEMPERATURE', np.float32, ('TIME','Y','X'), zlib=True, fill_value=-9999)
bottom_temp_adjusted = nc_out.createVariable('BOTTOM_TEMPERATURE_ADJUSTED', np.float32, ('TIME','Y','X'), zlib=True, fill_value=-9999)
bottom_lons = nc_out.createVariable('LONGITUDE', np.float32, ('Y','X'), zlib=True, fill_value=-9999)
bottom_lats = nc_out.createVariable('LATITUDE', np.float32, ('Y','X'), zlib=True, fill_value=-9999)

#Variable Attributes
bottom_lats.units = 'degree_north'
bottom_lats.long_name = 'Latitude'
bottom_lats.standard_name = 'latitude'
bottom_lons.units = 'degree_east'
bottom_lons.long_name = 'Longitude'
bottom_lons.standard_name = 'longitude'
times.units = 'seconds since 1900-01-01 00:00:00'
times.calendar = 'gregorian'
times.long_name = 'Time'
times.standard_name = 'time'
bottom_temp.units = 'degC'
bottom_temp.long_name = 'sea floor temperature'
bottom_temp.standard_name = 'bottom_temperature'
bottom_temp_adjusted.units = 'degC'
bottom_temp_adjusted.long_name = 'sea floor temperature - time adjusted'
bottom_temp_adjusted.standard_name = 'bottom_temperature_adjusted'
bottom_saln.units = 'psu'
bottom_saln.long_name = 'sea floor salinity'
bottom_saln.standard_name = 'bottom_salinity'
bottom_saln_adjusted.units = 'psu'
bottom_saln_adjusted.long_name = 'sea floor salinity - time adjusted'
bottom_saln_adjusted.standard_name = 'bottom_salinity_adjusted'

#Fill in the 2D,3D variables
bottom_saln[:,:,:] = saln
bottom_saln_adjusted[:,:,:] = saln_adj
bottom_temp[:,:,:] = temp
bottom_temp_adjusted[:,:,:] = temp_adj
bottom_lons[:,:] = longitude
bottom_lats[:,:] = latitude

#Fill in the dimensions
xs[:] = np.arange(longitude.shape[1])
ys[:] = np.arange(latitude.shape[0])
time_stamp = [pd.Timestamp(i) for i in ds.time.values]
times[:] = nc.date2num(time_stamp, units=times.units, calendar=times.calendar)

#Save and close the .nc file
nc_out.close()



#Create a csv output of the data as well 
#Gridded with latitude and longitude as the dimensions
#One file for each year of each season (44 x 3)

#Cycle through each year
for i,value in enumerate(time_stamp):

	#Isolate temperature
	df_temp = pd.DataFrame(
		temp[i],
		columns = longitude[0,:],index = latitude[:,0],
		)
	df_temp.index.name = 'latitude'
	df_temp.columns.name = 'longitude'
	df_temp.to_csv(finl_path+'csv_files/CABOTS_'+season+'_seafloortemperature_'+str(value.year)+'.csv')

	#Isolate salinity
	df_saln = pd.DataFrame(
		saln[i],
		columns = longitude[0,:],index = latitude[:,0],
		)
	df_saln.index.name = 'latitude'
	df_saln.columns.name = 'longitude'
	df_saln.to_csv(finl_path+'csv_files/CABOTS_'+season+'_seafloorsalinity_'+str(value.year)+'.csv')
	print(str(value.year)+' done!')

#Save the climatology of temperature and salinity as well
clim_years = np.arange(1991,2020+1)
path = '/home/jcoyne/Documents/Bottom_Stats/final_product/'
ds_clim = xr.open_dataset(path+'CABOTS_'+season+'.nc')
ds_clim = ds_clim.isel(TIME = np.isin(ds_clim['TIME.year'],clim_years))
clim_temp = ds_clim.BOTTOM_TEMPERATURE.mean(axis=0).values
clim_saln = ds_clim.BOTTOM_SALINITY.mean(axis=0).values
longitude = ds_clim.LONGITUDE.values
latitude = ds_clim.LATITUDE.values
df_temp = pd.DataFrame(
	clim_temp,
	columns = longitude[0,:],index = latitude[:,0],
	)
df_temp.index.name = 'latitude'
df_temp.columns.name = 'longitude'
df_temp.to_csv(finl_path+'csv_files/CABOTS_'+season+'_seafloortemperature_climatology'+str(clim_years[0])+'-'+str(clim_years[-1])+'.csv')
df_saln = pd.DataFrame(
	clim_saln,
	columns = longitude[0,:],index = latitude[:,0],
	)
df_saln.index.name = 'latitude'
df_saln.columns.name = 'longitude'
df_saln.to_csv(finl_path+'csv_files/CABOTS_'+season+'_seafloorsalinity_climatology'+str(clim_years[0])+'-'+str(clim_years[-1])+'.csv')


