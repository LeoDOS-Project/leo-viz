# Geographic Snapshots

The four `*.aoi-data.json.gz` files are prepared global sources, not AOI presets.
They are tracked with Git LFS. The browser fetches the chosen source from the
deployed site's `assets/aois/` directory on first generation and reuses it for
the session. The saved `default-jobs.json.gz` preset is unchanged.

## GDP

[Kummu et al., gridded GDP v4](https://doi.org/10.5281/zenodo.18429133), total
GDP PPP for 2024, band 35 of the 30 arc-minute raster, resampled onto the existing
1,440 by 720 equal-area grid. There are 264,199 positive cells. See [GDP.md](GDP.md)
for provenance and the source hash. GDP is a synthetic workload weighting proxy,
not a measurement of demand for satellite imagery. See the source record for
its authorship and license.

## Forests

[ESA WorldCover 2021 v200](https://doi.org/10.5281/zenodo.7254221), tree-cover
class 10, sampled from categorical COG overviews on a 0.05-degree grid. The
7,200 by 3,600 mask covers all 2,651 indexed source tiles. This is deliberately
coarse: cells are about 5.6 km north-south and do not estimate forest fraction.
Forests are potential monitoring regions, not wildfire detections.

Attribution: ESA WorldCover project 2021 / Contains modified Copernicus Sentinel
data (2021) processed by ESA WorldCover consortium. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Rivers

[RiverATLAS v1.0](https://www.hydrosheds.org/hydroatlas), all 10 global source
regions. The compact snapshot retains 857,210 reaches with discharge at least
10 cubic metres/second. Coordinates are simplified to 0.01 degrees, preserving
endpoints and downstream IDs. Connections to filtered reaches end the path;
small streams are absent.

Attribution: [Linke et al. (2019)](https://doi.org/10.1038/s41597-019-0300-6),
HydroATLAS / RiverATLAS, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source archive SHA-256:
`d8781793a2c2aae425417641a0e62b7f8f07efbba2b01357387acf898c82eae4`.

## Coasts

[Natural Earth land v5.1.1, 1:10 million](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/).
Land partitions are dissolved before extracting ocean exteriors. The 6,847
shoreline segments are simplified to 0.01 degrees, oriented with land on the
left, and exclude artificial map-boundary edges and inland lake holes. These
coasts have less detail than OpenStreetMap; custom OSM-derived networks remain
supported if supplied in the portable format.

Made with Natural Earth.
[Public domain](https://www.naturalearthdata.com/about/terms-of-use/).
Source archive SHA-256:
`e547d749445eaa0964aba76738090ec88f5e63c4585122170f98c67a7ea922dc`.

## Maintenance

Source metadata and raw input hashes are inside each snapshot. Existing prepared
caches can be packaged using the [Rust exporter](../../tools/leo-aoi/README.md).
The exporter does not ingest raw rasters or shapefiles. Deployments must hydrate
LFS objects and publish actual gzip files, not LFS pointer text. See the Pages
workflow for validation.
