import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata
import cmocean as cm
import shapely.geometry
import rasterio.features
import warnings
import time as tt
import pandas as pd
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import csv
import os
import skfuzzy as fuzz
from area import area
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapefile


'''
The purpose of this script is to average the bottom temperature in each of the regions.
This will be saved in a .csv output.
'''


#Import the shapefiles of interest
path_shp = '/home/jcoyne/Documents/Bottom_Stats/operation_files/NAFO_Divisions_SHP/'
shape_file = shapefile.Reader(path_shp + 'NAFO_Divisions_2021_poly_not_clipped.shp')

#Record all of the names
shape_file_names = []
for record in shape_file.records():
	shape_file_names.append(record.Label)

#Define the regions of interest
roi = ['0-A','0-B','1A','1B','1C','1D','1E','1F','2G','2H','2J','3K','3L','3M','3N','3O','3Pn','3Ps','4R','4S','4T','4Vn','4Vs','4W','4X','5Y','5Ze','5Zw']
regions = {}
for shape in shape_file.shapeRecords():
	if np.isin(shape.record.Label,roi):
		regions[shape.record.Label] = {'xlat': np.array(shape.shape.points)[:,1], 'xlon': np.array(shape.shape.points)[:,0]}

#Import the SFA divisions
myshp = open(os.path.expanduser('~/github/AZMP-NL/utils/SFAs/SFAs_PANOMICS_Fall2020_shp/SFAs_PANOMICS_Fall2020.shp'), 'rb')
mydbf = open(os.path.expanduser('~/github/AZMP-NL/utils/SFAs/SFAs_PANOMICS_Fall2020_shp/SFAs_PANOMICS_Fall2020.dbf'), 'rb')
r = shapefile.Reader(shp=myshp, dbf=mydbf, encoding = "ISO8859-1")
records = r.records()
shapes = r.shapes()
sfaoi = ['1','4','5','6','7']
#Fill dictionary with shapes
for idx, rec in enumerate(records):
	if rec[1] == 'Eastern Assessment Zone':
		regions['sfa2'] = {'xlat': np.array(shapes[idx].points)[:,1],'xlon': np.array(shapes[idx].points)[:,0]}
	elif rec[1] == 'Western Assessment Zone':
		regions['sfa3'] = {'xlat': np.array(shapes[idx].points)[:,1],'xlon': np.array(shapes[idx].points)[:,0]}
	elif np.isin(rec[0], sfaoi):
		regions['sfa'+rec[0]] = {'xlat': np.array(shapes[idx].points)[:,1],'xlon': np.array(shapes[idx].points)[:,0]}


#Import the bathymetry
path = '/home/jcoyne/Documents/Datasets/GEBCO_2023/'
ds_bath = xr.open_dataset(path + 'GEBCO_2023_sub_ice_topo.nc')

#Isolate the region of interest
lonLims = [-70,-42]
latLims = [39,80]
res = 30
ds_bath = ds_bath.where((ds_bath.lon>lonLims[0]) & (ds_bath.lon<lonLims[1]), drop=True)
ds_bath = ds_bath.where((ds_bath.lat>latLims[0]) & (ds_bath.lat<latLims[1]), drop=True)
lons_bath,lats_bath = np.meshgrid(ds_bath.lon.values[::res], ds_bath.lat.values[::res])
max_depth_bath = ds_bath.elevation[::res,::res].values*-1
lons_bath = lons_bath[:-1,:-1]
lats_bath = lats_bath[:-1,:-1]
max_depth_bath = max_depth_bath[:-1,:-1]

#Define the season and years that will be averaged
years = np.arange(1980,2024+1).astype(str)
season = 'fall'

#Import the climatology for the season in order to fill gaps
clim_years = np.arange(1990,2021+1)
path = '/home/jcoyne/Documents/Bottom_Stats/final_product/'
ds_clim = xr.open_dataset(path+'CABOTS_'+season+'.nc')
ds_clim = ds_clim.isel(TIME = np.isin(ds_clim['TIME.year'],clim_years))
clim_temp = ds_clim.BOTTOM_TEMPERATURE.mean(axis=0).values
clim_saln = ds_clim.BOTTOM_SALINITY.mean(axis=0).values
longitude = ds_clim.LONGITUDE.values
latitude = ds_clim.LATITUDE.values

#Create masks for each of the regions 
regional_mask = {}
for region in regions:

	#Create a polygon out of the coordinates
	polygon = Polygon(zip(regions[region]['xlon'], regions[region]['xlat']))

	#Cycle through and mask the data
	regional_mask[region] = np.full(longitude.shape, np.nan)
	for i in np.arange(longitude.shape[0]):
		for ii in np.arange(latitude.shape[1]):

			#Create the point of interest
			point = Point(longitude[i,ii],latitude[i,ii])
			if polygon.contains(point):
				regional_mask[region][i,ii] = 1
	print(region+' done!')

#Add the combined regions manually - where applicable include Ecosystem Production Units names
regional_mask['2HJ'] = regional_mask['2H'].copy()
regional_mask['2HJ'][regional_mask['2J'] == 1] = 1
regional_mask['2GH_labradorshelf'] = regional_mask['2G'].copy()
regional_mask['2GH_labradorshelf'][regional_mask['2H'] == 1] = 1
regional_mask['2J3K_newfoundlandshelf'] = regional_mask['2J'].copy()
regional_mask['2J3K_newfoundlandshelf'][regional_mask['3K'] == 1] = 1
regional_mask['3LNO_grandbanks'] = regional_mask['3L'].copy()
regional_mask['3LNO_grandbanks'][regional_mask['3N'] == 1] = 1
regional_mask['3LNO_grandbanks'][regional_mask['3O'] == 1] = 1
regional_mask['4RS'] = regional_mask['4R'].copy()
regional_mask['4RS'][regional_mask['4S'] == 1] = 1
regional_mask['4RST'] = regional_mask['4R'].copy()
regional_mask['4RST'][regional_mask['4S'] == 1] = 1
regional_mask['4RST'][regional_mask['4T'] == 1] = 1
regional_mask['4VWX_scotianshelf'] = regional_mask['4Vn'].copy()
regional_mask['4VWX_scotianshelf'][regional_mask['4Vs'] == 1] = 1
regional_mask['4VWX_scotianshelf'][regional_mask['4W'] == 1] = 1
regional_mask['4VWX_scotianshelf'][regional_mask['4X'] == 1] = 1

#Rename divisions with one only Ecosystem Production Unit
regional_mask['3M_flemishcap'] = regional_mask.pop('3M')
regional_mask['5Ze_georgesbank'] = regional_mask.pop('5Ze')
regional_mask['5Y_gulfofmaine'] = regional_mask.pop('5Y')


#Create the regional average dictionaries
regional_avg = {}
for region in regional_mask:
	regional_avg[region] = {}
	regional_avg[region]['Tmean'] = np.full(years.size, np.nan)
	regional_avg[region]['Tmean_sha100'] = np.full(years.size, np.nan)
	regional_avg[region]['Tmean_sha200'] = np.full(years.size, np.nan)
	regional_avg[region]['Tmean_sha300'] = np.full(years.size, np.nan)
	regional_avg[region]['Tmean_deep200'] = np.full(years.size, np.nan)
	regional_avg[region]['Tmean_deep300'] = np.full(years.size, np.nan)
	regional_avg[region]['area_colder0'] = np.full(years.size, np.nan)
	regional_avg[region]['area_colder1'] = np.full(years.size, np.nan)
	regional_avg[region]['area_warmer2'] = np.full(years.size, np.nan)
	regional_avg[region]['area_shrimp'] = np.full(years.size, np.nan)
	regional_avg[region]['area_colder2'] = np.full(years.size, np.nan)
	regional_avg[region]['area_colder2_perc'] = np.full(years.size, np.nan)
	regional_avg[region]['area_Pborealis'] = np.full(years.size, np.nan)
	regional_avg[region]['area_Pborealis_perc'] = np.full(years.size, np.nan)
	regional_avg[region]['area_Pmontagui'] = np.full(years.size, np.nan)
	regional_avg[region]['area_Pmontagui_perc'] = np.full(years.size, np.nan)
	regional_avg[region]['T_sampled_area'] = np.full(years.size, np.nan)
	regional_avg[region]['total_area'] = np.full(years.size, np.nan)
	if region == 'sfa2':
		regional_avg[region]['Pbor_eaz_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_eaz_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pbor_eaz_perc'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_eaz_perc'] = np.full(years.size, np.nan)
	if region == 'sfa3':
		regional_avg[region]['Pbor_waz_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_waz_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pbor_waz_perc'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_waz_perc'] = np.full(years.size, np.nan)
	if region == 'sfa4':
		regional_avg[region]['Pbor_sfa4_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_sfa4_habitat'] = np.full(years.size, np.nan)
		regional_avg[region]['Pbor_sfa4_perc'] = np.full(years.size, np.nan)
		regional_avg[region]['Pmon_sfa4_perc'] = np.full(years.size, np.nan)
	regional_avg[region]['T_percent_coverage'] = np.full(years.size, np.nan)
	regional_avg[region]['Smean'] = np.full(years.size, np.nan)
	regional_avg[region]['Smean_sha200'] = np.full(years.size, np.nan)
	regional_avg[region]['S_sampled_area'] = np.full(years.size, np.nan)
	regional_avg[region]['S_percent_coverage'] = np.full(years.size, np.nan)



#Import the data of interest
path = '/home/jcoyne/Documents/Bottom_Stats/final_product/'
ds = xr.open_dataset(path+'CABOTS_'+season+'.nc')

#Cycle through each of the years
for i,year in enumerate(years[:]):

	#Import the temperature and salinity
	bottom_temp = ds.BOTTOM_TEMPERATURE[i].values
	bottom_saln = ds.BOTTOM_SALINITY[i].values

	#Cycle through each of the regions and average
	for region in regional_mask:

		#Isolate the bathymetry (remove land pixels)
		bath_mask = max_depth_bath*regional_mask[region]
		bath_mask[bath_mask <= 10] = np.nan
		bath_mask[bath_mask > 1000] = np.nan

		#Determine the pixel area
		obj = {'type': 'Polygon', 'coordinates':
		[[[lons_bath[0,0],lats_bath[0,0]],
		[lons_bath[0,0],lats_bath[-1,0]],
		[lons_bath[-1,-1],lats_bath[-1,-1]],
		[lons_bath[0,-1],lats_bath[0,0]],
		[lons_bath[0,0],lats_bath[0,0]]]]}
		pixel_area = area(obj)/1e6/bottom_temp.size

		#Determine the percentage of missing area
		T_percent_coverage = np.sum(~np.isnan(bottom_temp*bath_mask))/np.sum(~np.isnan(bath_mask))
		T_percent_coverage = T_percent_coverage*100
		S_percent_coverage = np.sum(~np.isnan(bottom_saln*bath_mask))/np.sum(~np.isnan(bath_mask))
		S_percent_coverage = S_percent_coverage*100

		#Create the depth masks
		mask_100m = (bath_mask<=100).astype(float)
		mask_100m[mask_100m == 0] = np.nan
		mask_200m = (bath_mask<=200).astype(float)
		mask_200m[mask_200m == 0] = np.nan
		mask_300m = (bath_mask<=300).astype(float)
		mask_300m[mask_300m == 0] = np.nan
		mask_200m_dp = (bath_mask>=200).astype(float)
		mask_200m_dp[mask_200m_dp == 0] = np.nan
		mask_300m_dp = (bath_mask>=300).astype(float)
		mask_300m_dp[mask_300m_dp == 0] = np.nan

		#Fill in missing area with climatology
		bath_mask_bool = bath_mask.copy()
		bath_mask_bool[~np.isnan(bath_mask_bool)] = 1.
		btm_temp_region = bottom_temp*bath_mask_bool
		clm_temp_region = clim_temp*bath_mask_bool
		btm_temp_region[np.isnan(btm_temp_region)] = clm_temp_region[np.isnan(btm_temp_region)]
		btm_saln_region = bottom_saln*bath_mask_bool
		clm_saln_region = clim_saln*bath_mask_bool
		btm_saln_region[np.isnan(btm_saln_region)] = clm_saln_region[np.isnan(btm_saln_region)]

		#Mean temperature of entire area
		Tmean = np.nanmean(btm_temp_region)

		#Mean salinity of entire area
		Smean = np.nanmean(btm_saln_region)

		#Mean temperature at depths shallower than 100m, 200m, 300m
		Tmean100 = np.nanmean(btm_temp_region*mask_100m)
		Tmean200 = np.nanmean(btm_temp_region*mask_200m)
		Tmean300 = np.nanmean(btm_temp_region*mask_300m)
		Tmean200_dp = np.nanmean(btm_temp_region*mask_200m_dp)
		Tmean300_dp = np.nanmean(btm_temp_region*mask_300m_dp)

		#Mean salinity at depths shallower than 200m
		Smean200 = np.nanmean(btm_saln_region*mask_200m)

		#Area with temperature < 0
		area_colder_0deg = btm_temp_region[btm_temp_region <= 0].size*pixel_area
		#Area with temperature < 1
		area_colder_1deg = btm_temp_region[btm_temp_region <= 1].size*pixel_area
		#Area with temperature > 2
		area_warmer_2deg = btm_temp_region[btm_temp_region >= 1].size*pixel_area
		#Area with temperature > 2 & < 4 (shrimp habitat)
		area_shrimp = btm_temp_region[(btm_temp_region >= 2) & (btm_temp_region <= 4)].size*pixel_area
		#Area with temperature < 2 (crab habitat)
		area_colder_2deg = btm_temp_region[btm_temp_region <= 2].size*pixel_area
		#% area with temperature < 2 (crab habitat)
		area_colder_2deg_perc = area_colder_2deg/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100

		#Pandalus Borealis habitat
		Pbor = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 460) & (bath_mask >= 180) & (btm_temp_region >= -0.2) & (btm_temp_region <= 4.7)]))*pixel_area
		Pbor_perc = Pbor/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
		#Pandalus Montagui habitat
		Pmon = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 600) & (bath_mask >= 110) & (btm_temp_region >= -1) & (btm_temp_region <= 3.7)]))*pixel_area
		Pmon_perc = Pmon/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100

		#Record the sampled area and the total area
		T_sampled_area = np.sum(~np.isnan(btm_temp_region))*pixel_area
		S_sampled_area = np.sum(~np.isnan(btm_saln_region))*pixel_area
		total_polygon_area = np.sum(~np.isnan(bath_mask))*pixel_area

		#Area of NSRF seafloor with conditions within a certain depth and temperature range (project with Wojciech)
		if region == 'sfa2':
			Pbor_eaz = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 590) & (bath_mask >= 180) & (btm_temp_region >= -0.4) & (btm_temp_region <= 4.7)]))*pixel_area
			Pbor_eaz_perc = Pbor_eaz/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
			Pmon_eaz = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 600) & (bath_mask >= 120) & (btm_temp_region >= -0.5) & (btm_temp_region <= 3.7)]))*pixel_area
			Pmon_eaz_perc = Pmon_eaz/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
		if region == 'sfa3':
			Pbor_waz = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 520) & (bath_mask >= 210) & (btm_temp_region >= -0.7) & (btm_temp_region <= 4.0)]))*pixel_area
			Pbor_waz_perc = Pbor_waz/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
			Pmon_waz = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 530) & (bath_mask >= 110) & (btm_temp_region >= -1.2) & (btm_temp_region <= 2.8)]))*pixel_area
			Pmon_waz_perc = Pmon_waz/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
		if region == 'sfa4':
			Pbor_sfa4 = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 590) & (bath_mask >= 180) & (btm_temp_region >= -0.7) & (btm_temp_region <= 4.7)]))*pixel_area
			Pbor_sfa4_perc = Pbor_sfa4/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100
			Pmon_sfa4 = np.sum(~np.isnan(btm_temp_region[(bath_mask <= 590) & (bath_mask >= 140) & (btm_temp_region >= -0.9) & (btm_temp_region <= 4.0)]))*pixel_area
			Pmon_sfa4_perc = Pmon_sfa4/(np.sum(~np.isnan(btm_temp_region))*pixel_area)*100

		#Record all in the dictionary
		regional_avg[region]['Tmean'][i] = Tmean.round(3)
		regional_avg[region]['Tmean_sha100'][i] = Tmean100.round(3)
		regional_avg[region]['Tmean_sha200'][i] = Tmean200.round(3)
		regional_avg[region]['Tmean_sha300'][i] = Tmean300.round(3)
		regional_avg[region]['Tmean_deep200'][i] = Tmean200_dp.round(3)
		regional_avg[region]['Tmean_deep300'][i] = Tmean300_dp.round(3)
		regional_avg[region]['area_colder0'][i] = area_colder_0deg.round(3)
		regional_avg[region]['area_colder1'][i] = area_colder_1deg.round(3)
		regional_avg[region]['area_warmer2'][i] = area_warmer_2deg.round(3)
		regional_avg[region]['area_shrimp'][i] = area_shrimp.round(3)
		regional_avg[region]['area_colder2'][i] = area_colder_2deg.round(3)
		regional_avg[region]['area_colder2_perc'][i] = area_colder_2deg_perc.round(3)
		regional_avg[region]['area_Pborealis'][i] = Pbor.round(3)
		regional_avg[region]['area_Pborealis_perc'][i] = Pbor_perc.round(3)
		regional_avg[region]['area_Pmontagui'][i] = Pmon.round(3)
		regional_avg[region]['area_Pmontagui_perc'][i] = Pmon_perc.round(3)
		regional_avg[region]['T_sampled_area'][i] = T_sampled_area.round(3)
		regional_avg[region]['total_area'][i] = total_polygon_area.round(3)
		if region == 'sfa2':
			regional_avg[region]['Pbor_eaz_habitat'][i] = Pbor_eaz.round(3)
			regional_avg[region]['Pmon_eaz_habitat'][i] = Pmon_eaz.round(3)
			regional_avg[region]['Pbor_eaz_perc'][i] = Pbor_eaz_perc.round(3)
			regional_avg[region]['Pmon_eaz_perc'][i] = Pmon_eaz_perc.round(3)
		if region == 'sfa3':
			regional_avg[region]['Pbor_waz_habitat'][i] = Pbor_waz.round(3)
			regional_avg[region]['Pmon_waz_habitat'][i] = Pmon_waz.round(3)
			regional_avg[region]['Pbor_waz_perc'][i] = Pbor_waz_perc.round(3)
			regional_avg[region]['Pmon_waz_perc'][i] = Pmon_waz_perc.round(3)
		if region == 'sfa4':
			regional_avg[region]['Pbor_sfa4_habitat'][i] = Pbor_sfa4.round(3)
			regional_avg[region]['Pmon_sfa4_habitat'][i] = Pmon_sfa4.round(3)
			regional_avg[region]['Pbor_sfa4_perc'][i] = Pbor_sfa4_perc.round(3)
			regional_avg[region]['Pmon_sfa4_perc'][i] = Pmon_sfa4_perc.round(3)
		regional_avg[region]['T_percent_coverage'][i] = T_percent_coverage.round(3)
		regional_avg[region]['Smean'][i] = Smean.round(3)
		regional_avg[region]['Smean_sha200'][i] = Smean200.round(3)
		regional_avg[region]['S_sampled_area'][i] = S_sampled_area.round(3)
		regional_avg[region]['S_percent_coverage'][i] = S_percent_coverage.round(3)

		#Average and record
	print(year+' done!')

#Record the averages in a csv file
path_output = '/home/jcoyne/Documents/Bottom_Stats/final_product/csv_averages/'
for region in regional_avg:
	df = pd.DataFrame.from_dict(regional_avg[region], orient='index').transpose()
	df.index = years.astype(int)

	#Save all the data for internal purposes
	df.to_csv(path_output+'/internal/'+season+'_'+region+'_regional_averages.csv')

	#Remove some of the variables
	df = df.drop(['area_shrimp','area_Pborealis','area_Pborealis_perc','area_Pmontagui','area_Pmontagui_perc'],axis=1)
	if region == 'sfa2':
		df = df.drop(['Pbor_eaz_habitat','Pbor_eaz_perc'],axis=1)
	if region == 'sfa3':
		df = df.drop(['Pbor_waz_habitat','Pmon_waz_perc'],axis=1)
	if region == 'sfa4':
		df = df.drop(['Pbor_sfa4_habitat','Pbor_sfa4_perc'],axis=1)
	df.to_csv(path_output+'/'+season+'_'+region+'_regional_averages.csv')

#Record in pickle files as well (for Fred's scripts)
'''
pickle_files = ['3Ps','3LNO','3M','3K','3L','3O','2G','2H','2J','2HJ','2GH']
for i in pickle_files:
	dict_to_df = pd.DataFrame.from_dict(regional_avg[i], orient='columns')
	dict_to_df.index = years
	outname = path_output+season+'/'+'stats_'+i+'_'+season+'.pkl'
	dict_to_df.to_pickle(outname)
'''

