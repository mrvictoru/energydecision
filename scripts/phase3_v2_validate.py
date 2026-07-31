"""Quick validation: modern v2 model on SA1 Oct small battery, 2 RTG, both impacts."""
import sys, time; sys.path.insert(0, 'src')
import torch, json
from pathlib import Path
from decision_transformer import DecisionTransformer
from aemo_dt_hf import load_model_kwargs
from datetime import datetime
import polars as pl

DEVICE = 'cuda'
config = load_model_kwargs('configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json')
init={k:v for k,v in config.items() if k in {'state_dim','act_dim','n_block','h_dim','context_len','n_heads','drop_p','max_timestep','rope_enabled','rope_max_position','rope_base','n_kv_heads','qk_norm','tie_weights'}}
model=DecisionTransformer(**init)
ckpt=torch.load('models/aemo/dt/hf_v2_modern/aemo_dt_fcas_model.pt',map_location=DEVICE,weights_only=False)
model.load_from_checkpoint(ckpt); model.to(DEVICE); model.eval()
print(f"v2 loaded: {type(model).__name__} h_dim={model.h_dim} ctx={model.context_len} rs={getattr(model,'return_scale',None)}")

from aemo_data import fetch_aemo_dispatch_price, fetch_aemo_fcas_price, fetch_aemo_generation_by_fuel, build_supply_curve, aggregate_fcas_market_depth
from AEMOBatteryEnv import AEMODataPreprocessor, AEMOBatteryTradingEnv
from decision import AEMOAgent

start=datetime(2024,10,1); end=datetime(2024,10,14)
prices=fetch_aemo_dispatch_price(start,end,'SA1')
fl=[]
for svc in ['RAISE6SEC','RAISE60SEC','RAISE5MIN','RAISEREG','LOWER6SEC','LOWER60SEC','LOWER5MIN','LOWERREG']:
    d=fetch_aemo_fcas_price(start,end,'SA1',svc)
    if d.height>0: fl.append(d)
fcas=pl.concat(fl); gen=fetch_aemo_generation_by_fuel(start,end,'SA1')
prep=AEMODataPreprocessor(step_duration_hours=0.08332, add_normalized_features=True)
proc=prep.preprocess_aemo_data(prices,fcas,gen)
curves=build_supply_curve('SA1',start,end)
depth=aggregate_fcas_market_depth('SA1',start,end,demand_series=proc)
print(f"scenario: {proc.shape[0]} intervals")

bat=dict(capacity=8.0,max_flow=30.0,step_h=0.08333,init_soc=4.0)
for impact in ['identity','piecewise_merit_order']:
    for rtg in [0.0, 10.0]:
        env=AEMOBatteryTradingEnv(aemo_data=proc,battery_capacity=bat['capacity'],max_battery_flow=bat['max_flow'],
            step_duration=bat['step_h'],init_battery_level=bat['init_soc'],max_step=proc.shape[0],
            action_mode='full_fcas',degradation_mode='none',battery_life_cost=0.0,random_episode_start=False,
            impact_model=impact,impact_intensity=1.0,
            supply_curves=curves if impact!='identity' else None,
            fcas_depth=depth if impact!='identity' else None)
        agent=AEMOAgent(env,algorithm='dt',model=model,rtg_value=rtg)
        t0=time.time()
        ep_df,_=agent.run_episode()
        infos=ep_df['info'].to_list()
        energy=sum(i.get('energy_revenue',0) for i in infos)
        fcas_rev=sum(i.get('fcas_revenue',0) for i in infos)
        print(f"  {impact:>20} rtg={rtg:>4}: profit=${energy+fcas_rev:>9,.0f} E=${energy:>7,.0f} F=${fcas_rev:>7,.0f} ({time.time()-t0:.1f}s)")
