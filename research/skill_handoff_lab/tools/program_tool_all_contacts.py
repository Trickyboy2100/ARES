"""Read-only 100Hz tool-table/finger-tool instrumentation; unchanged control."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import program_tool_effect_probe as run
class AllContacts(run.ContactRecorder):
 def __init__(self,env):
  super().__init__(env);u=self.unwrapped;old=u._after_simulation_step
  def after():
   old();pairs={'tool_table':(u.l_shape_tool,u.scene.actors['table-workspace'])}
   for name in ['panda_hand','panda_leftfinger','panda_rightfinger']:
    link=u.agent.robot.links_map[name];pairs[name+'_tool']=(link,u.l_shape_tool);pairs[name+'_table']=(link,u.scene.actors['table-workspace'])
   self.substeps[-1]['additional_forces_N']={name:float(np.linalg.norm(run.serial(u.scene.get_pairwise_contact_forces(left,right)[0]))) for name,(left,right) in pairs.items()}
  u._after_simulation_step=after
run.ContactRecorder=AllContacts
if __name__=='__main__':
 try:run.main()
 finally:
  out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(exist_ok=True,parents=True);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(out/'CONTACT_PROVENANCE.json').write_text(json.dumps(dict(scope='read-only after100Hz simulation step; tool/table and hand/finger/tool/table contact forces',changes_physics=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
