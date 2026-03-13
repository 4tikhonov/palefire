import re
import pandas as pd
import os

content = """
[Download Highest Resolution of each Mandatory Field](https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz)
[Download Lowest Resolution of Each Mandatory Field](https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz)
[albedo_modis](https://www2.mmm.ucar.edu/wrf/src/wps_files/albedo_modis.tar.bz2)
[[greenfrac](https://www2.mmm.ucar.edu/wrf/src/wps_files/greenfrac.tar.bz2)_fpar_modis](https://www2.mmm.ucar.edu/wrf/src/wps_files/greenfrac_fpar_modis.tar.bz2)
[greenfrac_fpar_modis_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/greenfrac_fpar_modis_5m.tar.bz2)
[lai_modis_10m](https://www2.mmm.ucar.edu/wrf/src/wps_files/lai_modis_10m.tar.bz2)
[lai_modis_30s](https://www2.mmm.ucar.edu/wrf/src/wps_files/lai_modis_30s.tar.bz2)
[[maxsnowalb](https://www2.mmm.ucar.edu/wrf/src/wps_files/maxsnowalb.tar.bz2)_modis](https://www2.mmm.ucar.edu/wrf/src/wps_files/maxsnowalb_modis.tar.bz2)
[modis_landuse_20class_30s_with_lakes](https://www2.mmm.ucar.edu/wrf/src/wps_files/modis_landuse_20class_30s_with_lakes.tar.bz2)
[modis_landuse_20class_5m_with_lakes](https://www2.mmm.ucar.edu/wrf/src/wps_files/modis_landuse_20class_5m_with_lakes.tar.bz2)
[orogwd_2deg](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_2deg.tar.bz2)
[orogwd_1deg](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_1deg.tar.bz2)
[orogwd_30m](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_30m.tar.bz2)
[orogwd_20m](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_20m.tar.bz2)
[orogwd_10m](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_10m.tar.bz2)
[soiltemp_1deg](https://www2.mmm.ucar.edu/wrf/src/wps_files/soiltemp_1deg.tar.bz2)
[soiltype_bot_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/soiltype_bot_5m.tar.bz2)
[soiltype_bot_30s](https://www2.mmm.ucar.edu/wrf/src/wps_files/soiltype_bot_30s.tar.bz2)
[soiltype_top_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/soiltype_top_5m.tar.bz2)
[soiltype_top_30s](https://www2.mmm.ucar.edu/wrf/src/wps_files/soiltype_top_30s.tar.bz2)
[topo_gmted2010_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/topo_gmted2010_5m.tar.bz2)
[topo_gmted2010_30s](https://www2.mmm.ucar.edu/wrf/src/wps_files/topo_gmted2010_30s.tar.bz2)
[varsso](https://www2.mmm.ucar.edu/wrf/src/wps_files/varsso.tar.bz2)
[varsso_10m](https://www2.mmm.ucar.edu/wrf/src/wps_files/varsso_10m.tar.bz2)
[varsso_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/varsso_5m.tar.bz2)
[varsso_2m](https://www2.mmm.ucar.edu/wrf/src/wps_files/varsso_2m.tar.bz2)
[clayfrac_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/clayfrac_5m.tar.bz2)
[erod](https://www2.mmm.ucar.edu/wrf/src/wps_files/erod.tar.bz2)
[sandfrac_5m](https://www2.mmm.ucar.edu/wrf/src/wps_files/sandfrac_5m.bz2)
[crop](https://www2.mmm.ucar.edu/wrf/src/wps_files/crop.tar.bz2)
[groundwater](https://www2.mmm.ucar.edu/wrf/src/wps_files/groundwater.tar.bz2)
[soilgrids](https://www2.mmm.ucar.edu/wrf/src/wps_files/soilgrids.tar.bz2)
[nlcd2011_can_ll_9s](https://www2.mmm.ucar.edu/wrf/src/wps_files/nlcd2011_can_ll_9s.tar.bz2)
[NUDAPT44_1KM](https://www2.mmm.ucar.edu/wrf/src/wps_files/NUDAPT44_1KM.tar.bz2)
[ssib_landuse_10m](https://www2.mmm.ucar.edu/wrf/src/wps_files/ssib_landuse_10m.tar.bz2)
[lake_depth](https://www2.mmm.ucar.edu/wrf/src/wps_files/lake_depth.tar.bz2)
[bathymetry](https://www2.mmm.ucar.edu/wrf/src/wps_files/topobath_30s.tar.bz2)
[CGLC-MODIS-LCZ_100m](https://www2.mmm.ucar.edu/wrf/src/wps_files/cglc_modis_lcz_global.tar.gz)
[orogwd3_2deg](https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd3_2deg.tar.bz2)
[slucm-distributed-drag](https://www2.mmm.ucar.edu/wrf/src/wps_files/slucm_distributed_drag.tar.gz)
[albedo_ncep](https://www2.mmm.ucar.edu/wrf/src/wps_files/albedo_ncep.tar.bz2)
[landuse_30s_with_lakes](https://www2.mmm.ucar.edu/wrf/src/wps_files/landuse_30s_with_lakes.tar.bz2)
[bnu_soiltype_bot](https://www2.mmm.ucar.edu/wrf/src/wps_files/bnu_soiltype_bot.tar.bz2)
[bnu_soiltype_top](https://www2.mmm.ucar.edu/wrf/src/wps_files/bnu_soiltype_top.tar.bz2)
[modis_landuse_20class_15s](https://www2.mmm.ucar.edu/wrf/src/wps_files/modis_landuse_20class_15s.tar.bz2)
[modis_landuse_20class_15s_with_lakes](https://www2.mmm.ucar.edu/wrf/src/wps_files/modis_landuse_20class_15s_with_lakes.tar.gz)
[nlcd2006_ll_9s](https://www2.mmm.ucar.edu/wrf/src/wps_files/nlcd2006_ll_9s.tar.bz2)
[updated_Iceland_LU](https://www2.mmm.ucar.edu/wrf/src/wps_files/updated_Iceland_LU.tar.gz)
"""

# Regexp to match [Name](URL)
matches = re.findall(r'\[([^\]]+)\]\((https://[^\)]+)\)', content)

data = []
for name, url in matches:
    clean_name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', name)
    data.append({
        "title": clean_name, 
        "download_url": url, 
        "datasetContactEmail": "admin@example.org", 
        "dsDescriptionValue": f"Data file for {clean_name}"
    })

df = pd.DataFrame(data)
os.makedirs("cache", exist_ok=True)
df.to_csv("cache/wrf_inventory_v2.csv", index=False)
print(f"Generated cache/wrf_inventory_v2.csv with {len(df)} entries.")
