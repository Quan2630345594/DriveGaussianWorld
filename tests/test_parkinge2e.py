"""Focused preparation math tests; no ROS installation or bag required."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location("prepare_parkinge2e", Path(__file__).parents[1]/"scripts/prepare_parkinge2e.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class ParkingPreparationTests(unittest.TestCase):
    def test_vectorized_ocam_matches_scalar_row_column(self):
        o = dict(height=8,width=12,invpol=[10,2,-.5,.1],center_row_column=[4.1,6.2],affine_c_d_e=[1.01,.02,-.01])
        mx,my = prep.ocam_maps(o)
        for i,j in [(0,0),(2,9),(4,6),(7,11)]:
            row,col = i-4,j-6
            norm = np.hypot(row,col)
            if norm:
                theta = np.arctan((-12/4)/norm)
                rho = sum(coef*theta**k for k,coef in enumerate(o["invpol"]))
                x,y = row/norm*rho,col/norm*rho
                expected = [x*1.01+y*.02+4.1,x*(-.01)+y+6.2]
            else:
                expected = [4.1,6.2]
            np.testing.assert_allclose([my[i,j],mx[i,j]],expected,atol=1e-6)

    def test_official_optical_vehicle_transform(self):
        ov,vo = prep.extrinsics(dict(extrinsics=dict(yaw=90,pitch=0,roll=0,x=2,y=1,z=.5)))
        np.testing.assert_allclose(ov@vo,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(vo@np.array([0,0,0,1]),[2,1,.5,1],atol=1e-12)
        # Optical forward points along the camera body's +x, rotated to vehicle +y.
        np.testing.assert_allclose(vo@np.array([0,0,1,0]),[0,1,0,0],atol=1e-12)

    def test_interpolation_preserves_epoch_precision_and_yaw_anchor(self):
        epoch = 1708689770544374923
        ref = dict(time_ns=np.array([epoch,epoch+100_000_000],dtype=np.int64),
                   position_utm_m=np.array([[100,200,0],[100,202,0]]),
                   quaternion_xyzw=Rotation.from_euler("z",[90,100],degrees=True).as_quat(),
                   velocity_linear=np.array([[0,20,0],[0,20,0]]))
        p,q,v,d,g = prep.interpolate(ref,np.array([epoch+50_000_000],dtype=np.int64),np.array([100,200,0]),np.pi/2)
        np.testing.assert_allclose(p,[[1,0,0]],atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q).as_euler("ZYX")[:,0],np.deg2rad([5]),atol=1e-12)
        self.assertEqual(int(d[0]),50_000_000)
        self.assertEqual(int(g[0]),100_000_000)
        with self.assertRaises(RuntimeError):
            prep.interpolate(ref,np.array([epoch-1],dtype=np.int64),np.zeros(3),0)

    def test_json_refuses_overwrite(self):
        # Test against this source file so no test artifacts are created.
        with self.assertRaises(FileExistsError):
            prep.write_json(Path(__file__),{})


if __name__ == "__main__":
    unittest.main()
