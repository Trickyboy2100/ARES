"""Read-only per-link contact instrumentation of the unchanged tool controller.

No extra simulation steps or collision response changes. Compare complete poses
against earlier traces before attributing any differences to instrumentation.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import program_multiskill_pull as runner

class ContactRecorder(runner.Recorder):
    def __init__(self,env):
        self.substeps=[]
        super().__init__(env)
        u=self.unwrapped;original=u._after_simulation_step
        def after_physics():
            original()
            # BaseEnv fetches state only after the entire20Hz control step.
            # Explicit readback is required for genuine100Hz q/qvel/qacc.
            if u.gpu_sim_enabled:
                u.scene.px.gpu_fetch_articulation_qpos()
                u.scene.px.gpu_fetch_articulation_qvel()
                u.scene.px.gpu_fetch_articulation_qacc()
            pairs=dict(u.agent.robot.links_map);pairs['held_tool']=u.l_shape_tool
            forces={name:runner.serial(u.scene.get_pairwise_contact_forces(link,u.cube)[0]) for name,link in pairs.items()}
            self.substeps.append(dict(control_step=len(self.records)+1,substep=len(self.substeps)+1,phase=self.phase,robot_q=runner.serial(u.agent.robot.get_qpos()[0]),robot_qvel=runner.serial(u.agent.robot.get_qvel()[0]),robot_qacc=runner.serial(u.agent.robot.qacc[0]),force_vectors_N=forces,force_norms_N={k:float(np.linalg.norm(v)) for k,v in forces.items()},physics_dt_s=float(u.scene.px.timestep)))
        u._after_simulation_step=after_physics
    def step(self,action):
        self.substeps=[]
        return super().step(action)
    def measure(self):
        row=super().measure();u=self.unwrapped
        vectors={name:runner.serial(u.scene.get_pairwise_contact_forces(link,u.cube)[0]) for name,link in u.agent.robot.links_map.items()}
        row['robot_cube_force_vectors_N']=vectors
        row['robot_cube_force_norms_N']={k:float(np.linalg.norm(v)) for k,v in vectors.items()}
        row['contact_observation_valid']=self.phase!='RESET'
        row['physics_substep_contacts']=self.substeps.copy()
        return row

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',type=Path,required=True);args,_=p.parse_known_args()
    runner.Recorder=ContactRecorder
    try:runner.main()
    finally:
        if args.out.exists():
            source=Path(__file__);data=source.read_bytes();(args.out/source.name).write_bytes(data)
            manifest=args.out/'MANIFEST.json'
            if manifest.exists():
                m=json.loads(manifest.read_text());m['source_sha256'][source.name]=hashlib.sha256(data).hexdigest();m['contact_instrumentation']='all robot links and tool against cube; forces and actual robot q/qvel/qacc after each actual100Hzphysics substep; explicit read-only PhysX articulation fetch because native BaseEnv otherwise fetches at20Hz using existing _after_simulation_step hook, plus20Hz frame values; reset values invalid; no new physics steps';manifest.write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
