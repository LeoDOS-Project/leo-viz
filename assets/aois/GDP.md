# Bundled GDP Demand

`gdp.aoi-data.json.gz` contains 264,199 positive demand cells on the existing
1,440 by 720 spherical equal-area grid. It is an export of the same ingestion
configuration used by Default Jobs, not an AOI preset.

Source: Kummu et al., global gridded GDP v4, total GDP PPP for 2024, band 35 of
the 30 arc-minute raster: https://doi.org/10.5281/zenodo.18429133.
The source raster SHA-256 is
`29c32a5e7f96e6f07f4103ba9099e3181408619580c2ed30e9fe5264fea20f58`.
See the source record for authorship and licensing. Values are modelled GDP,
not measured Earth-observation demand. Dataset provenance is retained inside
the export.

Reproduce after the offline setup in `tools/leo-aoi/README.md`:

```sh
cd tools/leo-aoi
.venv/bin/python scripts/fetch_gdp.py
.venv/bin/python -m leo_aoi ingest examples/gdp.json data/gdp-demand.npz
.venv/bin/python -m leo_aoi.export data/gdp-demand.npz ../../assets/aois/gdp.aoi-data.json.gz
```
