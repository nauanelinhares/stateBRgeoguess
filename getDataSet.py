# Import necessary libraries

import requests
import os 

import pandas as pd
import numpy as np
import shapely

import geopandas as gpd
import concurrent.futures

# Define Class GoogleMapsView
class GoogleMapsView(object):
    def __init__(self, API_KEY, targetShapeFile, targetColumn) -> None:
        """
        Initialize GoogleMapsView object.

        Args:
            API_KEY (str): Key to access Google Maps API.
            targetShapeFile (str): SHP file for target of interest.
            targetColumn (str): Column name in the shapefile for target.
        """
        self.API_KEY = API_KEY
        self.targetShapeFile = targetShapeFile
        self.targetColumn = targetColumn
        self.targetCoords, self.target = self.getCoordsTarget()
        self.numOfErros = 0
        self.radius = 0
    
    def getCoordsTarget(self) -> np.array:
        """
        Get coordinates from target shapefile.

        Returns:
            np.array: Array of target coordinates.
        """
        shape = gpd.read_file(self.targetShapeFile)
        targetCoords = []
        target = []
        for i in range(len(shape['geometry'])):
            target.append(shape[self.targetColumn][i])
            if type(shape['geometry'][i]) == shapely.geometry.multipolygon.MultiPolygon:
                multi_polygon = shape['geometry'][i]
                x, y = multi_polygon.geoms[0].exterior.xy[0], multi_polygon.geoms[0].exterior.xy[1]
                targetCoords.append(np.array([(x,y) for x,y in zip(x,y)]))
            else:
                x, y = shape['geometry'][i].exterior.xy[0], shape['geometry'][i].exterior.xy[1]
                targetCoords.append(np.array([(x,y) for x,y in zip(x,y)]))
        return targetCoords, target
    
    def getImageParallel(self, radius, dadosPopulacao, i):
        """
        Get images in parallel using multiple threads.

        Args:
            radius (int): Radius for Street View image search.
            dadosPopulacao (pd.DataFrame): Population data.
            i (int): Index of the target.
        """
        self.radius = radius
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(self.getImageSingle, self.radius, dadosPopulacao, i) for _ in range(1000)]
            concurrent.futures.wait(futures)
    
    def getImageSingle(self, radius, dadosPopulacao, i):
        """
        Get a single Street View image.

        Args:
            radius (int): Radius for Street View image search.
            dadosPopulacao (pd.DataFrame): Population data.
            i (int): Index of the target.
        """
        # Get random latitude and longitude from region limits
        files = os.listdir(f'dataSet/{self.target[i]}')
        target_files = [file for file in files if file.startswith(self.target[i])]
        n = len(target_files)
   
        if  n<1600:
            error = True
            dadosPopulacao = dadosPopulacao[(dadosPopulacao['latitude']>=self.targetCoords[i][:,1].min()) & 
                                                (dadosPopulacao['latitude']<=self.targetCoords[i][:,1].max()) & 
                                                (dadosPopulacao['longitude']>=self.targetCoords[i][:,0].min()) &
                                                (dadosPopulacao['longitude']<=self.targetCoords[i][:,0].max())].reset_index(drop=True)
            
            dadosPopulacao['populacao'] = dadosPopulacao['populacao'] / dadosPopulacao['populacao'].sum()
            des_lat = np.std(dadosPopulacao['latitude'])
            des_long = np.std(dadosPopulacao['longitude'])
            while error or n<1600:
                lat = np.random.choice(dadosPopulacao['latitude'], p=dadosPopulacao['populacao'])
                lat = np.random.uniform(lat-des_lat/len(dadosPopulacao), lat + des_lat/len(dadosPopulacao))
                long = np.random.choice(dadosPopulacao['longitude'], p=dadosPopulacao['populacao'])
                long = np.random.uniform(long-des_long/len(dadosPopulacao), long+des_long/len(dadosPopulacao))

                # Verify if there is an image at this point
                try:
                    url = f'https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{long}&radius={self.radius}&key={self.API_KEY}'
                    response = requests.get(url, timeout=1).json()
                    if response['status'] == 'OK':
                        print(n, self.target[i], self.radius, self.numOfErros)
                        pano_id = response["pano_id"]
                        files = os.listdir(f'dataSet/{self.target[i]}')
                        target_files = [file for file in files if file.endswith(f'{pano_id}.tiff')]
                        if len(target_files) == 0:
                            url = f'https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={pano_id}&cb_client=search.revgeo_and_fetch.gps&w=600&h=300&yaw=354.7943&pitch=0&thumbfov=100&quot'
                            response = requests.get(url, timeout=4)
                            if response.status_code == 200:
                                response = response.content
                                files = os.listdir(f'dataSet/{self.target[i]}')
                                target_files = [file for file in files if file.startswith(self.target[i])]
                                n = len(target_files)
                                with open(f'dataSet/{self.target[i]}/{self.target[i]}_{n+1}_{pano_id}.tiff', 'wb') as handler:
                                    handler.write(response)
                                error = False
                                self.numOfErros = 0
                    else:
                        self.numOfErros += 1
                        if self.numOfErros == 100:
                            self.radius = round(self.radius * 1.2)
                            self.numOfErros = 0
                            print(self.target[i], ': ' ,'new_radius', self.radius)
                except:
                    pass

# Set to True if municipality data is not available
get_data_municipios = False

if get_data_municipios:
    # Get data from IBGE
    response =  requests.get('https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2021/variaveis/9324?localidades=N6[all]').json()
    dadosPopulacao = pd.DataFrame(response[0]['resultados'][0]['series'])

    # Get name for each locality
    dadosPopulacao['nome'] = dadosPopulacao['localidade'].apply(lambda x: x['nome'])
    dadosPopulacao['id'] = dadosPopulacao['localidade'].apply(lambda x: x['id'])
    dadosPopulacao['populacao'] = dadosPopulacao['serie'].apply(lambda x: x['2021'])

    # Load municipality data
    muni = gpd.read_file('BR_Municipios_2022.shp')

    # Create column with latitude and longitude
    dadosPopulacao['latitude'] = dadosPopulacao['id'].apply(lambda x: muni.loc[muni['CD_MUN'] == x]['geometry'].values[0].centroid.xy[1][0])
    dadosPopulacao['longitude'] = dadosPopulacao['id'].apply(lambda x: muni.loc[muni['CD_MUN'] == x]['geometry'].values[0].centroid.xy[0][0])
    
    # Save municipality data to CSV
    dadosPopulacao.to_csv('dadosPopulacao.csv', index=False)
    
# Load municipality data
dadosPopulacao = pd.read_csv('dadosPopulacao.csv')

API_KEY = "YOUR_API_KEY"
view = GoogleMapsView(API_KEY, 'BR_UF_2022.shp', 'SIGLA_UF')

for i in range(len(view.target)):
    try:
        files = os.listdir(f'dataSet/{view.target[i]}')
    except:
        os.mkdir(f'dataSet/{view.target[i]}')
        files = os.listdir(f'dataSet/{view.target[i]}')
    target_files = [file for file in files if file.startswith(view.target[i])]
    n = len(target_files)
    print(view.target[i], n)
    if  n<1600:
        view.getImageParallel(100, dadosPopulacao, i)