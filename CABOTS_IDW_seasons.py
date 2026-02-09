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
The purpose of this script is to create monthly NetCDF files for bottom temperature over a specified time period.
These will later be used to create a monthly bottom temperature climatology.
'''

#Import the bathymetry that the data will be interpolated to
path = '/home/coynej/Documents/Datasets/GEBCO_2023/'
ds_bath = xr.open_dataset(path + 'gebco_2025_n90.0_s0.0_w-120.0_e0.0.nc')

#Isolate the region of interest
lonLims = [-100,-42]
latLims = [35,80]

#lonLims = [-75,-65]
#latLims = [37,45]

ds_bath = ds_bath.where((ds_bath.lon>lonLims[0]) & (ds_bath.lon<lonLims[1]), drop=True)
ds_bath = ds_bath.where((ds_bath.lat>latLims[0]) & (ds_bath.lat<latLims[1]), drop=True)

#Create an array
res = 30 #makes a 0.125 deg resolution
lons_bath,lats_bath = np.meshgrid(ds_bath.lon.values[::res], ds_bath.lat.values[::res])
max_depth_bath = ds_bath.elevation[::res,::res].values*-1

#Create a land mask polygon list
land_mask = max_depth_bath <= 0 

#Convert this mask into a shapely
shapes = rasterio.features.shapes(land_mask.astype('uint8'))
polygons_land = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes if shape[1] == 1]

#Cycle through each and change the coordinate system
polygons_land_latlon = []
area_land = []
for i,polygon in enumerate(polygons_land):
	#Remove polygons that are very small
	if polygon.area > 10:
		#Convert the points to lat lon coordinates
		x,y = polygon.exterior.xy
		x_lon = np.array(x,dtype=int)
		y_lat = np.array(y,dtype=int)
		x_lon[x_lon == lons_bath.shape[1]] = lons_bath.shape[1]-1
		y_lat[y_lat == lats_bath.shape[0]] = lats_bath.shape[0]-1
		x_lon = lons_bath[0,:][x_lon]
		y_lat = lats_bath[:,0][y_lat]
		#Create a new polygon
		polygons_land_latlon.append(shapely.geometry.Polygon(np.array((x_lon,y_lat)).T))
		area_land.append(shapely.geometry.Polygon(np.array((x_lon,y_lat)).T).area)
polygons_land_latlon = np.array(polygons_land_latlon)[np.argsort(area_land)[::-1]]

'''
bathymetry_depth = max_depth_bath.copy()
bathymetry_lon = lons_bath.copy()
bathymetry_lat = lats_bath.copy()
insitu_temperature = temp.copy()
insitu_salinity = saln.copy()
insitu_month_T = temp_months.copy()
insitu_month_S = saln_months.copy()
#levels
insitu_lon = lons.copy()
insitu_lat = lats.copy()
land_polygons = polygons_land_latlon.copy()
power=2
max_distance=2
min_distance_landcheck=0.2
membership_function='linear'
window_size=25
'''


#Function to determine bottom temperature and salinity
def IDW_bottom_var(
	interpolated_depths,
	bathymetry_depth,
	bathymetry_lon,
	bathymetry_lat,
	insitu_temperature,
	insitu_salinity,
	insitu_month_T,
	insitu_month_S,
	levels,
	insitu_lon,
	insitu_lat,
	land_polygons,
	window_size=25,
	max_distance=2,
	power=2,
	min_distance_landcheck=0.2,
	membership_function='linear',
	):
	'''
	Returns 2D bottom temperature array.
	interpolated_depths: array of depths which the bottom temperature will be interpolated at
	bathymetry_depth: 2D array of bathymetry depths (depth positive, land is 0 or negative)
	bathymetry_lon: 2D array of bathymetry longitude (same shape as bathymetry_depth)
	bathymetry_lat: 2D array of bathymetry latitude (same shape as bathymetry_depth)
	insitu_temperature: 2D array (time by level) of temperature
	insitu_lon: 1D array of longitude (time)
	insitu_lat: 1D array of latitude (time)
	land_polygons: list of polygons shapely that outline where the land is
	max_distance: the maximum distance insitu measurements can be from the interpolated point (degrees) (default 2)
	power: the power of the IDW (default 2)
	min_distance_landcheck: the minimum distance between interpolated point and insitu measurement where a land check will be done (default 0.2)
	membership_function: choose between linear or triangular (default linear)
	window_size: how far (+/-) to look around depth of interest for measurements (default 25)
	'''

	#Record the interpolated bottom variables
	bottom_temp_interp = np.full(bathymetry_depth.shape, np.nan)
	bottom_saln_interp = np.full(bathymetry_depth.shape, np.nan)
	bottom_mthT_interp = np.full(bathymetry_depth.shape, np.nan)
	bottom_mthS_interp = np.full(bathymetry_depth.shape, np.nan)

	R = np.full(bathymetry_depth.shape, np.nan)
	mu_R = np.full(bathymetry_depth.shape, np.nan)
	power = np.full(bathymetry_depth.shape, np.nan)

	#Determine which profiles are used from CASTS
	CASTS_index = np.arange(insitu_temperature.shape[0])
	CASTS_used = []

	#Cycle through each latitude and longitude
	for i in np.arange(bathymetry_lat.shape[0]):
		for ii in np.arange(bathymetry_lon.shape[1]):

			#Determine if the gridcell is within the depth range
			depth = bathymetry_depth[i,ii]
			if depth < interpolated_depths[0] or depth > interpolated_depths[-1]:
				continue

			#Isolate the latitude and longitude of interest
			lon_point = bathymetry_lon[i,ii]
			lat_point = bathymetry_lat[i,ii]

			#Determine the distances of each measurement (in-situ) to point
			distance = (lon_point - insitu_lon)**2 + (lat_point - insitu_lat)**2
			distance = np.sqrt(distance)

			#Isolate the lon,lat,bottom vars within max_distance
			lon_slice = insitu_lon[distance <= max_distance]
			lat_slice = insitu_lat[distance <= max_distance]
			temp_slice = insitu_temperature[distance <= max_distance,:]
			saln_slice = insitu_salinity[distance <= max_distance,:]
			mthT_slice = insitu_month_T[distance <= max_distance,:]
			mthS_slice = insitu_month_S[distance <= max_distance,:]
			distance_slice = distance[distance <= max_distance]
			CASTS_index_slice = CASTS_index[distance <= max_distance]

			#Isolate variables within range of measurement
			#Take the closest measurement to depth within +/-window_size depth range
			temp_slice = temp_slice[:,(levels>depth-window_size)*(levels<depth+window_size)]
			saln_slice = saln_slice[:,(levels>depth-window_size)*(levels<depth+window_size)]
			mthT_slice = mthT_slice[:,(levels>depth-window_size)*(levels<depth+window_size)]
			mthS_slice = mthS_slice[:,(levels>depth-window_size)*(levels<depth+window_size)]

			#Cycle through each time and select the temperature closest to
			window_filt = ~np.isnan(temp_slice)*np.arange(temp_slice.shape[1])
			window_filt = np.argmin(np.abs(window_filt - window_size),axis=1)
			temps = np.array([temp_slice[iii][window_filt[iii]] for iii in np.arange(window_filt.size)])
			mthTs = np.array([mthT_slice[iii][window_filt[iii]] for iii in np.arange(window_filt.size)])
			insitu_mask_T = ~np.isnan(temps)

			#Cycle through each time and select the salinity closest to
			window_filt = ~np.isnan(saln_slice)*np.arange(saln_slice.shape[1])
			window_filt = np.argmin(np.abs(window_filt - window_size),axis=1)
			salns = np.array([saln_slice[iii][window_filt[iii]] for iii in np.arange(window_filt.size)])
			mthSs = np.array([mthS_slice[iii][window_filt[iii]] for iii in np.arange(window_filt.size)])
			insitu_mask_S = ~np.isnan(salns)

			#Isolate the lon,lat,bottom variables within max_distance
			lon_slice = lon_slice[insitu_mask_T+insitu_mask_S]
			lat_slice = lat_slice[insitu_mask_T+insitu_mask_S]
			temp_slice = temps[insitu_mask_T+insitu_mask_S]
			saln_slice = salns[insitu_mask_T+insitu_mask_S]
			mthT_slice = mthTs[insitu_mask_T+insitu_mask_S]
			mthS_slice = mthSs[insitu_mask_T+insitu_mask_S]
			distance_slice = distance_slice[insitu_mask_T+insitu_mask_S]
			CASTS_index_slice = CASTS_index_slice[insitu_mask_T+insitu_mask_S]

			#Determine if enough measurements are available
			if temp_slice.size < 3:

				#If any distance is less than 0.05, fill with mean of those
				if distance_slice.size > 0 and distance_slice.min() <= 0.05:
					bottom_temp_interp[i,ii] = temp_slice[distance_slice <= 0.05].mean()
					bottom_saln_interp[i,ii] = saln_slice[distance_slice <= 0.05].mean()
					bottom_mthT_interp[i,ii] = mthT_slice[distance_slice <= 0.05].mean()
					bottom_mthS_interp[i,ii] = mthS_slice[distance_slice <= 0.05].mean()
				continue

			#If there are 3 or more measurements, determine if any points are behind barriers (land)
			barrier_mask = np.full(lon_slice.size, False)

			#Determine the minimum distance of the interpolation point to land
			interpolation_Point = shapely.geometry.Point(lon_point,lat_point)
			Point_distance = []
			for polygon in polygons_land_latlon:
				Point_distance.append(interpolation_Point.distance(polygon))
			Point_distance = np.min(Point_distance)

			#If the point is more than max_distance from land, skip land_mass check
			if Point_distance <= max_distance:

				#Cycle through each of the measurements being considered
				for iii in np.arange(lon_slice.size):

					#Determine if the measurement is very close, if so, skip barrier check
					if distance_slice[iii] < min_distance_landcheck:
						barrier_mask[iii] = True
					else:

						#Create a line shapely
						line = shapely.geometry.LineString([[lon_point,lat_point],[lon_slice[iii],lat_slice[iii]]])

						#Determine if the line intersects any of the polygons
						land_warning = False
						for polygon in land_polygons:
							if line.intersects(polygon):
								land_warning = True
								break

						#If a False value is present, skip the measurement
						if land_warning == False:
							barrier_mask[iii] = True

				#Re-isolate the values
				lon_slice = lon_slice[barrier_mask]
				lat_slice = lat_slice[barrier_mask]
				temp_slice = temp_slice[barrier_mask]
				saln_slice = saln_slice[barrier_mask]
				mthT_slice = mthT_slice[barrier_mask]
				mthS_slice = mthS_slice[barrier_mask]
				distance_slice = distance_slice[barrier_mask]
				CASTS_index_slice = CASTS_index_slice[barrier_mask]

			#Determine if enough measurements are available
			if temp_slice.size < 3:

				#If any distance is less than 0.05, fill with mean of those
				if distance_slice.size > 0 and distance_slice.min() <= 0.05:
					bottom_temp_interp[i,ii] = temp_slice[distance_slice <= 0.05].mean()
					bottom_saln_interp[i,ii] = saln_slice[distance_slice <= 0.05].mean()
					bottom_mthT_interp[i,ii] = mthT_slice[distance_slice <= 0.05].mean()
					bottom_mthS_interp[i,ii] = mthS_slice[distance_slice <= 0.05].mean()
				continue

			#Cycle through each of the variables
			interpolated_values = []
			for var in [temp_slice,saln_slice,mthT_slice,mthS_slice]:

				#Take away nan values
				nan_filt = ~np.isnan(var)
				lon_place = lon_slice[nan_filt]
				lat_place = lat_slice[nan_filt]
				var_place = var[nan_filt]
				dst_place = distance_slice[nan_filt]

				#Check to make sure data is present
				if var_place.size == 0:
					interpolated_values.append(np.nan)
					continue

				#Determine the power
				A = np.pi*(max_distance**2)
				r_exp = 1/(2*(var_place.size/A)**0.5)
				r_obs = np.mean(dst_place)
				R[i,ii] = r_obs/r_exp
				R_min = 1
				R_max = 8
				if R[i,ii] < R_min:
					mu_R[i,ii] = 0
				elif R[i,ii] > R_max:
					mu_R[i,ii] = 1
				else:
					mu_R[i,ii] = fuzz.smf(np.array([0,R[i,ii]]),R_min,R_max)[-1]
				if membership_function == 'linear':
					if mu_R[i,ii] <= 0.1:
						power[i,ii] = 0.1
					elif mu_R[i,ii] >= 0.9:
						power[i,ii] = 3
					else:
						slope = (3-0.1)/(0.9-0.1)
						b = 0.1-slope*0.1
						power[i,ii] = slope*mu_R[i,ii]+b

				#Find the interpolated value
				numerator = np.sum(var_place / (dst_place ** power[i,ii]))
				weights = np.sum(1 / (dst_place ** power[i,ii]))
				interpolated_value = numerator/ weights
				interpolated_values.append(interpolated_value)

			#Record all the interpolated values
			bottom_temp_interp[i,ii] = interpolated_values[0]
			bottom_saln_interp[i,ii] = interpolated_values[1]
			bottom_mthT_interp[i,ii] = interpolated_values[2]
			bottom_mthS_interp[i,ii] = interpolated_values[3]

			#Record the profiles used
			CASTS_filt = np.isin(CASTS_index_slice, CASTS_used)
			CASTS_used.extend(list(CASTS_index_slice[~CASTS_filt]))

	#Isolate the profiles of interest
	CASTS_used = np.array(CASTS_used)

	return(bottom_temp_interp,bottom_mthT_interp,bottom_saln_interp,bottom_mthS_interp,CASTS_used)


#Perform the IDW at each depth
interpolated_depths = np.arange(10,1000)

#State which season you want to isolate
season = 'fall'

#Define the years of interest
years = np.arange(1980,2025+1).astype(str)
for year in years[:]:

	#Import the CASTS file of interest
	path = '~/data/CASTS/'
	ds = xr.open_dataset(path+year+'.nc')

	# Select time (save several options here)
	if season == 'summer':
		#ds = ds.sel(time=ds['time.season']=='JJA')
		ds_month = ds.sel(time=((ds['time.month']>=7)) & ((ds['time.month']<=9)))
	elif season == 'spring':
		#ds = ds.sel(time=ds['time.season']=='MAM')
		ds_month = ds.sel(time=((ds['time.month']>=4)) & ((ds['time.month']<=6)))
	elif season == 'fall':
		#ds = ds.sel(time=ds['time.season']=='SON')
		ds_month = ds.sel(time=((ds['time.month']>=10)) & ((ds['time.month']<=12)))
	else:
		print('!! no season specified, used them all! !!')

	#Define the maximum depth
	zmax = 1000

	# Isolate for temperatures of interest
	ds_month = ds_month.sel(level=ds_month['level']<zmax)
	temp = np.array(ds_month.temperature)
	saln = np.array(ds_month.salinity)

	#We are not interested in salinity below a certain threshold? Maybe 20psu?
	saln[saln < 20] = np.nan

	lons = np.array(ds_month.longitude)
	lats = np.array(ds_month.latitude)
	levels = np.array(ds_month.level)
	months = np.array(ds_month['time.month'])
	temp_months = np.full(temp.shape, np.nan)
	for i,value in enumerate(months):
		temp_months[i,~np.isnan(temp[i])] = value
	saln_months = np.full(saln.shape, np.nan)
	for i,value in enumerate(months):
		saln_months[i,~np.isnan(saln[i])] = value

	#Perform the bottom temperature interpolation using IDW
	start = tt.time()
	inter_temp,inter_mthT,inter_saln,inter_mthS,CASTS_used = IDW_bottom_var(
		interpolated_depths,
		max_depth_bath,
		lons_bath,
		lats_bath,
		temp,
		saln,
		temp_months,
		saln_months,
		levels,
		lons,
		lats,
		polygons_land_latlon,
		)
	end = tt.time()

	#Save the CASTS_output
	path_output = '/home/coynej/Documents/Bottom_Stats/climatology/'+season+'/'
	CASTS_used = np.array([lons[CASTS_used],lats[CASTS_used]]).T
	np.save(path_output+'CASTS_profilesused_'+season+'_'+year,CASTS_used)

	#Save the results into a netcdf
	#Set up the .nc file
	nc_out = nc.Dataset(path_output+year+'.nc','w')

	#File information
	nc_out.title = 'Bottom Temperature Salinity Seasonal Output' #Temporary title for the .nc file
	nc_out.institution = 'Northwest Atlantic Fisheries Centre, Fisheries and Oceans Canada'
	nc_out.description = 'Output by jonathan.coyne@dfo-mpo.gc.ca'
	nc_out.history = 'Created ' + tt.ctime(tt.time())

	#Create dimensions
	time = nc_out.createDimension('time', None) #use date2 for this
	x = nc_out.createDimension('x', inter_temp.shape[1])
	y = nc_out.createDimension('y', inter_temp.shape[0])

	#Create coordinate variables
	times = nc_out.createVariable('time', np.float64, ('time',))
	xs = nc_out.createVariable('x', np.int32, ('x',))
	ys = nc_out.createVariable('y', np.int32, ('y',))

	#Create 2D variables
	bottom_temp = nc_out.createVariable('bottom_temperature', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_monthT = nc_out.createVariable('bottom_monthT', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_saln = nc_out.createVariable('bottom_salinity', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_monthS = nc_out.createVariable('bottom_monthS', np.float32, ('time','y','x'), zlib=True, fill_value=-9999)
	bottom_lons = nc_out.createVariable('longitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)
	bottom_lats = nc_out.createVariable('latitude', np.float32, ('y','x'), zlib=True, fill_value=-9999)

	#Variable Attributes
	bottom_lats.units = 'degree_north'
	bottom_lons.units = 'degree_east'
	times.units = 'seconds since 1900-01-01 00:00:00'
	times.calendar = 'gregorian'

	#Fill in the 2D structure
	bottom_temp[:,:,:] = inter_temp[None,:,:]
	bottom_monthT[:,:,:] = inter_mthT[None,:,:]
	bottom_saln[:,:,:] = inter_saln[None,:,:]
	bottom_monthS[:,:,:] = inter_mthS[None,:,:]
	bottom_lons[:,:] = lons_bath[:,:]
	bottom_lats[:,:] = lats_bath[:,:]

	#Fill in the dimension variables
	xs[:] = np.arange(inter_temp.shape[1])
	ys[:] = np.arange(inter_temp.shape[0])
	time_stamp = pd.Timestamp(year=int(year),month=1,day=1)
	times[:] = nc.date2num(time_stamp, units=times.units, calendar=times.calendar)

	#Save and close the .nc file
	nc_out.close()
	print(year+' done! Time to complete - '+str(np.round((end-start)/60,2))+' minutes')




























































