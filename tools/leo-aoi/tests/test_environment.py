from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from shapely.geometry import Polygon, shape, box
from leo_aoi.environment import Config, Forest, Network, grid_polygon, rectangles_from_polygon, prepare
from leo_aoi.environment_data import save, segment_lengths
from leo_aoi.geometry import Aoi, Grid
from leo_aoi.viewer import write_preview

class EnvironmentTests(unittest.TestCase):
    def test_short_segments_near_dateline_use_double_precision_distances(self):
        points=np.array([[179,0],[180,0]],dtype="float32")
        self.assertAlmostEqual(float(segment_lengths(points)[0]),111.19492664455873,places=8)

    def test_invalid_configs_fail_before_download(self):
        for values in [{"count":0}, {"kind":"unknown"}, {"buffer_km":0},
                       {"compactness":2}, {"max_latitude":90}, {"mean_length_km":float("nan")},
                       {"seed":-1}, {"extra":1}]:
            with self.assertRaises(ValueError): Config.from_dict(values)

    def test_polygon_tiles_preserve_holes_and_wrap_dateline(self):
        grid=Grid.with_tile_edge(10)
        geographic=Polygon([(179,-1),(181,-1),(181,1),(179,1)],
                           [[(179.5,-.5),(180.5,-.5),(180.5,.5),(179.5,.5)]])
        rects=rectangles_from_polygon(grid_polygon(geographic,grid,180),grid)
        aoi=Aoi("seam",179.8,0,1,None,rects)
        tiles=list(aoi.tiles())
        self.assertEqual(len(tiles),len(set(tiles)))
        self.assertTrue(all(0<=r.col<r.col+r.width<=grid.columns for r in rects))
        geometry=shape(aoi.geojson(grid)["geometry"])
        self.assertTrue(geometry.is_valid)
        self.assertEqual(geometry.geom_type,"MultiPolygon")
        self.assertNotIn(grid.cell(180,0),set(tiles))

    def network(self, gap=False):
        points=np.array([[0,0],[.1,0],[.1,0],[.2,0]],dtype=float)
        if gap: points[2:,0]+=10
        lengths=np.array([segment_lengths(points[:2]).sum(),segment_lengths(points[2:]).sum()])
        return {"points":points,"offsets":np.array([0,2,4]),"next_index":np.array([1,-1]),
                "length":lengths,"flow":np.array([10,10]),"ids":np.array([100,101])}

    def test_network_crosses_real_junction_and_stops_at_outlet(self):
        config=Config(kind="river",mean_length_km=100,length_ratio=1)
        network=Network(self.network(),config)
        result=network.sample(np.random.default_rng(7),np.random.default_rng(2),config,Grid.with_tile_edge(.1))
        lon,lat,requested,rects,extra=result
        self.assertGreater(len(rects),0)
        self.assertTrue(extra["path_truncated"])
        self.assertLess(extra["path_length_km"],23)
        self.assertGreater(extra["path_length_km"],0)
        # A broken next-reach link must not create a thousand-kilometre bridge.
        network=Network(self.network(gap=True),config)
        result=network.sample(np.random.default_rng(7),np.random.default_rng(2),config,Grid.with_tile_edge(.1))
        self.assertLess(result[-1]["path_length_km"],12)

    def test_latitude_band_clips_sampling_length_instead_of_dropping_crossing_lines(self):
        points=np.array([[0,-2],[0,2]],dtype=float)
        data={"points":points,"offsets":np.array([0,2]),"next_index":np.array([-1]),
              "length":np.array([segment_lengths(points).sum()]),"flow":np.array([10]),"ids":np.array([1])}
        config=Config(kind="river",max_latitude=1,mean_length_km=1,length_ratio=1)
        network=Network(data,config)
        self.assertAlmostEqual(network.weights[-1]/data["length"][0],.5)
        for seed in range(10):
            result=network.sample(np.random.default_rng(seed),np.random.default_rng(seed),config,Grid.with_tile_edge(1))
            self.assertLessEqual(abs(result[1]),1)
        for values in [{"kind":"river","mean_length_km":1000,"length_ratio":100}]:
            with self.assertRaises(ValueError): Config(**values)

    def test_coast_buffer_respects_land_on_left(self):
        config=Config(kind="coast",mean_length_km=5,length_ratio=1,inland_km=3,offshore_km=.5)
        result=Network(self.network(),config).sample(np.random.default_rng(7),np.random.default_rng(2),config,Grid.with_tile_edge(.1))
        lon,lat,requested,rects,extra=result
        aoi=Aoi("coast",lon,lat,requested,None,rects)
        west,south,east,north=shape(aoi.geojson(Grid.with_tile_edge(.1))["geometry"]).bounds
        self.assertGreater(north,abs(south)*3)

    def test_closed_coastline_completes_one_circuit_from_any_start(self):
        points=np.array([[0,0],[.1,0],[.1,.1],[0,.1],[0,0]],dtype=float)
        length=float(segment_lengths(points).sum())
        data={"points":points,"offsets":np.array([0,5]),"next_index":np.array([0]),
              "length":np.array([length]),"flow":np.array([0]),"ids":np.array([1])}
        config=Config(kind="coast",mean_length_km=100,length_ratio=1)
        network=Network(data,config)
        for seed in range(3):
            result=network.sample(np.random.default_rng(seed),np.random.default_rng(1),config,Grid.with_tile_edge(.5))
            self.assertAlmostEqual(result[-1]["path_length_km"],length,delta=.001)

    def test_length_weighting_is_independent_of_record_count(self):
        data=self.network()
        data["length"]=np.array([1.0,9.0])
        network=Network(data,Config(kind="river"))
        rng=np.random.default_rng(42)
        picks=np.searchsorted(network.weights,rng.random(10000)*network.weights[-1])
        self.assertAlmostEqual(float((picks==1).mean()),.9,delta=.02)

    def test_forest_sampling_uses_ground_area_and_stays_bounded(self):
        mask=np.zeros((720,1440),dtype="uint8")
        mask[100:110,500:510]=1
        mask[350:360,500:510]=1
        forest=Forest({"mask":mask},Config())
        high=forest.weights[109]-forest.weights[99]
        low=forest.weights[359]-forest.weights[349]
        self.assertGreater(low,high*2)
        config=Config(mean_area_km2=1000,size_ratio=1)
        result=forest.sample(np.random.default_rng(2),np.random.default_rng(3),config,Grid.with_tile_edge(2))
        lon,lat,requested,rects,extra=result
        aoi=Aoi("forest",lon,lat,requested,None,rects)
        self.assertLess(aoi.tile_count*Grid.with_tile_edge(2).cell_area_km2,3000)
        self.assertTrue(shape(aoi.geojson(Grid.with_tile_edge(2))["geometry"]).is_valid)

    def test_viewer_round_trip_replay_and_prefix_with_local_source(self):
        with tempfile.TemporaryDirectory() as directory:
            save(Path(directory)/"river.npz",{"schema_version":1,"kind":"river"},**self.network())
            request={"schema_version":1,"environment":{"kind":"river","count":3,
                       "data_dir":directory,"mean_length_km":5,"length_ratio":1}}
            first=StringIO();write_preview(request,first)
            again=StringIO();write_preview(request,again)
            self.assertEqual(first.getvalue(),again.getvalue())
            output=json.loads(first.getvalue())
            self.assertEqual(output["config"]["kind"],"river")
            for aoi in output["aois"]:
                self.assertTrue(shape(aoi["geometry"]).is_valid)
                self.assertEqual(aoi["source_kind"],"river")
            longer=StringIO();write_preview({**request,"environment":{**request["environment"],"count":5}},longer)
            self.assertEqual(output["aois"],json.loads(longer.getvalue())["aois"][:3])

if __name__=="__main__":unittest.main()
