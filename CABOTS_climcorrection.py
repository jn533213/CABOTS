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
The purpose of this script is to apply the climate correction to the seasonal data
'''

#Import the data of interest
season = 'spring'
path = '/home/coynej/Documents/Bottom_Stats/climatology/'+season+'/'

#Define the climatology months of interest
path_clim = '/home/coynej/Documents/Bottom_Stats/climatology/'
if season == 'spring':
	start_month = '04' #should be in a 00 01 format
	end_month = '06' #same as above
	chosen_month = 7
elif season == 'summer':
	start_month = '07' #should be in a 00 01 format
	end_month = '09' #same as above
	chosen_month = 8
elif season == 'fall':
	start_month = '10' #should be in a 00 01 format
	end_month = '12' #same as above
	chosen_month = 11

#Cycle through the years of interest
years = np.arange(1980,2025+1).astype(str)
for year in years[:]:

	#Import the data of interest
	ds = xr.open_dataset(path+year+'.nc')

	#Start with temperature and salinity
	bottom_monthT = ds.bottom_monthT[0].values - chosen_month
	bottom_temp = ds.bottom_temperature[0].values
	bottom_monthS = ds.bottom_monthS[0].values - chosen_month
	bottom_saln = ds.bottom_salinity[0].values
	ds_climstart =  xr.open_dataset(path_clim+'bottomclim_'+start_month+'.nc')
	ds_climend =  xr.open_dataset(path_clim+'bottomclim_'+end_month+'.nc')

	#Determine the trend 
	clim_slope_T = (ds_climend.bottom_temperature.values - ds_climstart.bottom_temperature.values)/(int(end_month) - int(start_month))
	clim_slope_S = (ds_climend.bottom_salinity.values - ds_climstart.bottom_salinity.values)/(int(end_month) - int(start_month))

	#Determine the temperature and salinity adjustment
	bottom_nudge_T = bottom_monthT*clim_slope_T*-1
	bottom_nudge_S = bottom_monthS*clim_slope_S*-1

	#For bottom month values within 0.5 months of the target, don't perform nudge
	bottom_nudge_T[(bottom_monthT>=-0.5)*(bottom_monthT<=0.5)] = 0
	bottom_nudge_S[(bottom_monthS>=-0.5)*(bottom_monthS<=0.5)] = 0

	#Determine the final temperature and salinity
	bottom_temp_final = bottom_temp + bottom_nudge_T
	bottom_saln_final = bottom_saln + bottom_nudge_S


	#Save the results into a netcdf
	path_output = '/home/coynej/Documents/Bottom_Stats/adjusted/'+season+'/'

	#Set up the .nc file
	nc_out = nc.Dataset(path_output+year+'.nc','w')

	#File information
	nc_out.title = 'Bottom Temperature Salinity - Adjusted' #Temporary title for the .nc file
	nc_out.institution = 'Northwest Atlantic Fisheries Centre, Fisheries and Oceans Canada'
	nc_out.description = 'Output by jonathan.coyne@dfo-mpo.gc.ca'
	nc_out.history = 'Created ' + tt.ctime(tt.time())

	#Create dimensions
	time = nc_out.createDimension('time', None) #use date2 for this
	x = nc_out.createDimension('x', bottom_temp_final.shape[1])
	y = nc_out.createDimension('y', bottom_temp_final.shape[0])

	#Create coordinate variables
	times = nc_out.createVariable('time', np.float64, ('time',))
	xs = nc_out.createVariable('x', np.int32, ('x',))
	ys = nc_out.createVariable('y', np.int32, ('y',))

	#Create 2D variables
	bottom_temps = nc_out.createVariable('bottom_temperature', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_temp_adjusted = nc_out.createVariable('bottom_temperature_adjusted', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_slope_Ts = nc_out.createVariable('bottom_slope_T', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_nudge_Ts = nc_out.createVariable('bottom_nudge_T', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_salns = nc_out.createVariable('bottom_salinity', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_saln_adjusted = nc_out.createVariable('bottom_salinity_adjusted', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_slope_Ss = nc_out.createVariable('bottom_slope_S', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_nudge_Ss = nc_out.createVariable('bottom_nudge_S', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_lons = nc_out.createVariable('longitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	bottom_lats = nc_out.createVariable('latitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)

	#Variable Attributes
	bottom_lats.units = 'degree_north'
	bottom_lons.units = 'degree_east'
	times.units = 'seconds since 1900-01-01 00:00:00'
	times.calendar = 'gregorian'

	#Fill in the 2D structure
	bottom_temps[:,:,:] = bottom_temp[None,:,:]
	bottom_temp_adjusted[:,:,:] = bottom_temp_final[None,:,:]
	bottom_slope_Ts[:,:,:] = clim_slope_T[None,:,:]
	bottom_nudge_Ts[:,:,:] = bottom_nudge_T[None,:,:]
	bottom_salns[:,:,:] = bottom_saln[None,:,:]
	bottom_saln_adjusted[:,:,:] = bottom_saln_final[None,:,:]
	bottom_slope_Ss[:,:,:] = clim_slope_S[None,:,:]
	bottom_nudge_Ss[:,:,:] = bottom_nudge_S[None,:,:]
	bottom_lons[:,:] = ds.longitude.values[:,:]
	bottom_lats[:,:] = ds.latitude.values[:,:]

	#Fill in the dimension variables
	xs[:] = np.arange(bottom_temp_final.shape[1])
	ys[:] = np.arange(bottom_temp_final.shape[0])
	time_stamp = pd.Timestamp(year=int(year),month=1,day=1)
	times[:] = nc.date2num(time_stamp, units=times.units, calendar=times.calendar)

	#Save and close the .nc file
	nc_out.close()
	print(year+' done!')






'''
#Create a figure showing the following:
#Original temperature, slope, nudge, and adjusted temperature

#Import the original and adjusted temperature files
season = 'summer'
chosen_month = 8
year = '2023'
path = '/home/jcoyne/Documents/Bottom_Stats/climatology/temperature_'+season+'/'
ds_org = xr.open_dataset(path+year+'.nc')
path = '/home/jcoyne/Documents/Bottom_Stats/temperature_adjusted/'+season+'/'
ds_adj = xr.open_dataset(path+year+'.nc')

#Import the bathymetry that will be used for contour lines
path = '/home/jcoyne/Documents/Datasets/GEBCO_2023/'
ds_bath = xr.open_dataset(path + 'GEBCO_2023_sub_ice_topo.nc')

#Isolate the region of interest
#lonLims = [-65,-42]
#latLims = [40,60]
lonLims = [-70,-35]
latLims = [38,68]
ds_bath = ds_bath.where((ds_bath.lon>lonLims[0]) & (ds_bath.lon<lonLims[1]), drop=True)
ds_bath = ds_bath.where((ds_bath.lat>latLims[0]) & (ds_bath.lat<latLims[1]), drop=True)

#Create an array
lons_bath,lats_bath = np.meshgrid(ds_bath.lon.values[::10], ds_bath.lat.values[::10])
max_depth_bath = ds_bath.elevation[::10,::10].values*-1


#Create the figure
#Set up the map
land_10m = cfeature.NaturalEarthFeature('physical','land','10m',facecolor='tan')

#Cycle through each of the variables
for i in np.arange(4):

	#Create the subplot
	fig = plt.figure(figsize=(5,8))
	ax = plt.subplot(projection=ccrs.Mercator())

	#Plot the coastline
	ax.set_facecolor('white')
	ax.add_feature(land_10m,zorder=2)
	ax.set_extent([lonLims[0],lonLims[1],latLims[0],latLims[1]])

	#Plot the bottom temperature
	if i == 0:
		c = ax.pcolormesh(
			lons_bath,
			lats_bath,
			ds_org.bottom_month[0].values-chosen_month,
			vmin=-1,vmax=1,
			cmap=cm.cm.balance,zorder=1,
			transform=ccrs.PlateCarree()
			)
		plt.title('Bottom Temperature - Month',fontsize=8)
	elif i == 1:
		c = ax.pcolormesh(
			lons_bath,
			lats_bath,
			ds_adj.bottom_slope[0].values,
			vmin=-3,vmax=3,
			cmap=cm.cm.balance,zorder=1,
			transform=ccrs.PlateCarree()
			)
		plt.title('Bottom Temperature - Slope',fontsize=8)
	elif i == 2:
		c = ax.pcolormesh(
			lons_bath,
			lats_bath,
			ds_adj.bottom_nudge[0].values,
			vmin=-3,vmax=3,
			cmap=cm.cm.balance,zorder=1,
			transform=ccrs.PlateCarree()
			)
		plt.title('Bottom Temperature - Nudge',fontsize=8)
	elif i == 3:
		c = ax.pcolormesh(
			lons_bath,
			lats_bath,
			ds_adj.bottom_temperature[0].values,
			vmin=-2,vmax=6,
			cmap=cm.cm.thermal,zorder=1,
			transform=ccrs.PlateCarree()
			)
		plt.title('Bottom Temperature - Adjusted',fontsize=8)

	#Plot the bathymetry
	plt.contour(
		lons_bath,
		lats_bath,
		max_depth_bath,
		levels=[100,500,1000,4000],
		colors='grey',
		linewidths=0.75,
		zorder=2,
		transform=ccrs.PlateCarree()
		)

	#Add gridlines
	gl = ax.gridlines(draw_labels=['left','bottom'],xlocs=[-50,-55,-60],ylocs=[40,45,50,55,60],
		dms=True,x_inline=False,y_inline=False,linestyle='--')
	gl.xlabel_style = {'size': 8}
	gl.ylabel_style = {'size': 8}

	#Add a colour bar
	ax_coords = ax.get_position()
	cax = fig.add_axes([ax_coords.x0,ax_coords.y0-0.035,ax_coords.x1-ax_coords.x0,0.0125])
	cb = plt.colorbar(c, cax=cax, orientation='horizontal')
	#cb.set_label(r'$\rm T (^{\circ}C)$', fontsize=8, fontweight='normal')
	cb.ax.tick_params(labelsize=8)

	#Save the figure
	plt.savefig('/home/jcoyne/Documents/Bottom_Stats/figures/temperature_adjusted/'+season+\
		'/tempadjusted_'+year+'_'+str(i)+'.png', dpi=300)
	plt.close('all')
'''