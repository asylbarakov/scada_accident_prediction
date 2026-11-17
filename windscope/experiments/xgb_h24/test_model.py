import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model import RESEARCH_FEATURES,seven_features,fit_xgb,export_bundle,NativeAdapter


class XGBTests(unittest.TestCase):
    def raw(self):
        n=8
        raw=pd.DataFrame({'timestamp':pd.date_range('2024-01-01',periods=n,freq='10min',tz='UTC'),'asset_id':['a']*n})
        for i,name in enumerate(['sensor_0_avg','sensor_1_avg','sensor_43_avg','sensor_11_avg','wind_speed_3_avg','sensor_50','sensor_32_avg','sensor_33_avg','sensor_34_avg']):raw[name]=np.arange(n)+10*i
        raw['label']=1
        return raw

    def test_exact_feature_order_voltage_and_delay(self):
        raw=self.raw();features=seven_features(raw)
        self.assertEqual(features.columns.tolist(),RESEARCH_FEATURES)
        self.assertTrue(features.iloc[0].isna().all())
        self.assertEqual(features.loc[2,'voltage'],71.)
        self.assertEqual(features.loc[2,'sensor_0_avg'],1.)
        pd.testing.assert_frame_equal(seven_features(raw.iloc[:5]),features.iloc[:5])
        raw.loc[5:,'sensor_0_avg']=1e10;raw['label']=0
        pd.testing.assert_frame_equal(seven_features(raw).iloc[:5],features.iloc[:5])

    def test_gap_or_missing_phase_does_not_use_old_or_partial_voltage(self):
        raw=self.raw();raw.loc[1,'sensor_33_avg']=np.nan
        self.assertTrue(pd.isna(seven_features(raw).loc[2,'voltage']))
        raw=raw.drop(index=3).reset_index(drop=True)
        self.assertTrue(seven_features(raw).iloc[3].isna().all())
        raw=self.raw().iloc[::-1]
        with self.assertRaises(ValueError):seven_features(raw)

    def frame(self):
        return pd.DataFrame(dict(turbine_id=['a']*8+['b']*8+['c']*8,label=[0,0,0,1]*6,history_ready=[True]*24,
            x=np.tile([0.,1.,2.,3.,4.,5.,6.,7.],3))).assign(x=lambda f:f.x.where(f.turbine_id.ne('c'),1e6))

    def parameters(self):
        return dict(n_estimators=3,max_depth=2,learning_rate=.3,n_jobs=1,random_state=42,objective='binary:logistic',tree_method='hist',base_score=.5)

    def test_scaler_and_class_balance_use_only_train(self):
        model,scaler,info=fit_xgb(self.frame(),['x'],['a','b'],self.parameters())
        self.assertEqual(scaler.data_min_[0],0.)
        self.assertEqual(scaler.data_max_[0],7.)
        self.assertEqual(info['scale_pos_weight'],3.)
        self.assertEqual(info['training_rows'],16)
        self.assertEqual(model.n_features_in_,1)

    def test_native_bundle_round_trip_and_checksum_guard(self):
        frame=self.frame();model,scaler,_=fit_xgb(frame,['x'],['a','b'],self.parameters())
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(model,scaler,['x'],.5,{},Path(directory))
            adapter=NativeAdapter(Path(directory))
            expected=model.predict_proba(scaler.transform(frame[['x']].astype('float32')))[:,1]
            np.testing.assert_array_equal(adapter.predict(frame)['score'],expected)
            np.testing.assert_array_equal(adapter.predict(frame.drop(columns=['label']))['score'],expected)
            with self.assertRaises(ValueError):adapter.predict(frame.drop(columns=['x']))
            (Path(directory)/'booster.ubj').write_bytes(b'changed')
            with self.assertRaises(ValueError):NativeAdapter(Path(directory))


if __name__=='__main__':unittest.main()
