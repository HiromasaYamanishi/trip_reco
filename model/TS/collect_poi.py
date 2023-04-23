from pyrosm import get_data
import pyrosm
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os

if __name__=='__main__':
    fp = get_data('kanto')
    mesh_2020_01_all_df = pd.read_csv('/home/yamanishi/project/trip_recommend/data/flow/mesh/2020_01_mesh.csv')
    mesh1km_tokyo_2020_01_dir = Path('/home/yamanishi/project/trip_recommend/data/flow/13/2020/01')
    attribute_dir = Path('/home/yamanishi/project/trip_recommend/data/flow/attribute')
    attr_2020_df = pd.read_csv(attribute_dir/'attribute_mesh1km_2020.csv')
    mesh_2020_01_tokyo_df = pd.read_csv(mesh1km_tokyo_2020_01_dir/'monthly_mdp_mesh1km.csv')
    tokyo_meshid = mesh_2020_01_tokyo_df['mesh1kmid'].unique()
    tokyo_2020_01 = mesh_2020_01_all_df[mesh_2020_01_all_df['mesh1kmid'].isin(tokyo_meshid)].reset_index()

    data_dir = '/home/yamanishi/project/trip_recommend/TS/data/poi'
    for index in tqdm(range(len(tokyo_2020_01))):
        osm = pyrosm.OSM(fp, bounding_box=[tokyo_2020_01.loc[index, 'lon_min'], tokyo_2020_01.loc[index, 'lat_min'], tokyo_2020_01.loc[index, 'lon_max'], tokyo_2020_01.loc[index, 'lat_max']])
        meshid = tokyo_2020_01.loc[index, 'mesh1kmid']
        pois = osm.get_pois()
        filepath = os.path.join(data_dir, f'{meshid}.csv')
        if pois is not None:
            pois.to_csv(filepath)
        


