import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
import pickle, json, time

from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from AEMOBatteryEnv import AEMOBatteryTradingEnv

device = 'cuda'
manifest = json.loads(Path('models/aemo/dt/sdp_teacher_dt_loss_surface_manifest.json').read_text())
model = DecisionTransformer(**manifest['model_kwargs'])
model.load_from_checkpoint('models/aemo/dt/sdp_teacher_dt_best.pt', map_location=device)
model.to(device)
model.eval()

with open('/tmp/scenario_cache/sa1_oct_2024.pkl', 'rb') as f:
    sd = pickle.load(f)
processed = sd['processed']

env = AEMOBatteryTradingEnv(
    aemo_data=processed, battery_capacity=8.0, max_battery_flow=30.0,
    step_duration=0.08333, init_battery_level=4.0, max_step=processed.shape[0],
    action_mode='full_fcas', degradation_mode='real_world',
    degradation_chemistry='LFP', degradation_temperature=30.0,
    random_episode_start=False, impact_model='piecewise_merit_order',
    impact_intensity=1.0, supply_curves=sd['curves'], fcas_depth=sd['depth'],
)
agent = AEMOAgent(env, algorithm='dt', model=model, rtg_value=0.0)
t0 = time.time()
df, _ = agent.run_episode()
dt = time.time() - t0
infos = df['info'].to_list()
energy = sum(i.get('energy_revenue',0) for i in infos)
fcas = sum(i.get('fcas_revenue',0) for i in infos)
deg = sum(i.get('degradation_cost',0) for i in infos)
print(f'elapsed={dt:.0f}s rows={df.height} profit=${energy+fcas-deg:,.0f} E=${energy:,.0f} F=${fcas:,.0f} D=${deg:,.0f}')
