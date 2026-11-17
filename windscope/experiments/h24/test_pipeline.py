import unittest
import numpy as np
import pandas as pd
from pipeline import physical_features, assert_lineage_separation, fit_candidate


class LeakageTests(unittest.TestCase):
    def config(self):
        return dict(cadence_minutes=10,measurement_delay_steps=1,windows_hours=[1,6,24],required_history_hours=6,operation=None,sensors={'temperature':{'column':'sensor_x_avg','unit':'C','measurement':'temperature'}})

    def test_features_are_prefix_invariant_and_ignore_labels_and_future(self):
        config=self.config()
        raw=pd.DataFrame(dict(turbine_id=['a']*300,timestamp=pd.date_range('2024-01-01',periods=300,freq='10min',tz='UTC'),temperature=np.arange(300,dtype=float),label=[1]*300))
        full,ready=physical_features(raw,config)
        prefix,prefix_ready=physical_features(raw.iloc[:200],config)
        pd.testing.assert_frame_equal(full.iloc[:200],prefix)
        pd.testing.assert_series_equal(ready.iloc[:200],prefix_ready)
        changed=raw.copy();changed.loc[200:,'temperature']=1e10;changed['label']=0
        altered,_=physical_features(changed,config)
        pd.testing.assert_frame_equal(full.iloc[:200],altered.iloc[:200])
        self.assertEqual(full.loc[199,'temperature__now'],198.)
        broken=raw.drop(index=100).reset_index(drop=True)
        _,broken_ready=physical_features(broken,config)
        self.assertFalse(broken_ready.iloc[100:136].any())

    def test_repeated_physical_content_cannot_cross_turbines(self):
        frame=pd.DataFrame(dict(turbine_id=['a','b'],avg_fingerprint=['same','same'],full_fingerprint=['x','y']))
        with self.assertRaisesRegex(ValueError,'cross-turbine'):assert_lineage_separation(frame)
        frame['turbine_id']='a';assert_lineage_separation(frame)

    def test_scaler_and_imputer_fit_training_turbines_only(self):
        frame=pd.DataFrame(dict(turbine_id=['a']*4+['b']*2,label=[0,1,0,1,0,1],history_ready=[True]*6,x=[0.,2.,4.,np.nan,10000.,20000.]))
        config={'models':{'logistic':{'C':1.,'class_weight':'balanced','max_iter':3000}},'seed':42}
        model=fit_candidate('logistic',config,frame,['x'],['a'])
        self.assertEqual(model.named_steps['imputer'].statistics_[0],2.)
        self.assertEqual(model.named_steps['scaler'].mean_[0],2.)


if __name__=='__main__':unittest.main()
