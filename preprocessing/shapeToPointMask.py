import rasterio 
from shapely.geometry import box, Point
from shapely import get_coordinates
from rasterio.mask import  geometry_mask
import geopandas as gpd
import glob, os

image_dir = '../sample_datasets/image'
shape_path = '../sample_datasets/annotation.geojson'
mask_out_path = '../sample_datasets/mask'

gdf = gpd.read_file(shape_path)
gdf = gpd.GeoSeries(gdf.geometry.apply(lambda geom : [Point(xy) for xy in get_coordinates(geom)[:-1]]).explode()).reset_index(drop=True)

for image_path in glob.glob(os.path.join(image_dir, '*.tif')):

    filename = os.path.basename(image_path)
    with rasterio.open(image_path, 'r') as src:
        bbox = box(*src.bounds)
        image = src.read()
        points = gdf[bbox.contains(gdf)].values
        if len(points) == 0:
            print(image_path, '<= image has no annotation !')
            continue
        
        mask = geometry_mask(points, out_shape=image[0,:,:].shape, transform=src.transform, invert=True) * 255
        out_profile = {
            **src.profile,
            'count' : 1,
        }
        with rasterio.open(os.path.join(mask_out_path, filename), 'w', **out_profile) as dst:
            dst.write(mask, 1)
            
    print('completed =>', image_path)