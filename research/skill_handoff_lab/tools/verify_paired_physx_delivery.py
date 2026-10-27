"""Read-only hashes/AST/cold-start/media checks for the actual video package."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def physical_methods(path):
    tree=ast.parse(path.read_text());env=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Env')
    return {n.name:n for n in env.body if isinstance(n,ast.FunctionDef)}


def main(a):
    package=read(a.release/'VIDEO_PROVENANCE.json');checked=[];runs={}
    for record in package['comparisons']:
        directory=Path(record['mp4_local_path']).parent
        for name,meta in record['files'].items():
            path=directory/name;assert path.exists() and sha(path)==meta['sha256'],path
        if (directory/'composition_source.py').exists():assert sha(directory/'composition_source.py')==record['script_sha256']
        media=read_media(Path(record['mp4_local_path']));assert media['codec_name']=='h264' and (media['width'],media['height'])==(1920,1080) and media['r_frame_rate']=='30/1'
        for runpath in [record['left_run'],record['right_run']]:
            run=Path(runpath)
            if runpath in runs:continue
            assert (run/'RUN_COMPLETE.json').exists()
            capture=read(run/'VIDEO_CAPTURE_PROVENANCE.json');assert capture['state_writes_during_actions']==0 and capture['render_does_not_step_physics']
            assert sha(run/'actual_physics_trace.npz')==capture['physics_trace_sha256']
            assert sha(run/'action_log.json')==capture['action_log_sha256']
            assert sha(run/'paired_video_hook_source.py')==capture['hook_sha256']
            assert read(run/'APPEARANCE_IDENTITY.json')['equal']
            before=physical_methods(run/'base_task_source.py');after=physical_methods(run/'recorded_task.py')
            for name in ['_pre_physics_step','_apply_action','_reset_idx','_get_observations','state','physical_snapshot','_get_rewards','_get_dones']:
                if name in ['_get_dones','_get_observations']:
                    after[name].body=[n for n in after[name].body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='paired_video')]
                assert ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False),(run,name)
            camera_clock={}
            for case,meta in capture['cases'].items():
                assert sha(run/meta['mp4'])==meta['mp4_sha256']
                frames=read(run/meta['frame_records']);assert len(frames)==meta['frames']
                assert frames[0]['capture_event']=='INITIAL' and frames[-1]['capture_event']=='TERMINAL'
                assert all(x['elapsed_physics_s']<=y['elapsed_physics_s'] for x,y in zip(frames,frames[1:]))
                assert int(read_media(run/meta['mp4'])['nb_frames'])==len(frames)
                camera_clock[case]={'source_frame_count':len(frames),'last_physical_time_s':frames[-1]['elapsed_physics_s'],'max_encoded_frame_clock_minus_physical_time_s':max(abs(r['frame_index']/30-r['elapsed_physics_s']) for r in frames),'authoritative_clock':'elapsed_physics_s per captured frame; extra event frames are explicitly resampled into paired video'}
            runs[runpath]={'trace_sha256':capture['physics_trace_sha256'],'appearance_equal':True,'physical_method_AST_identical':True,'recorded_cases':sorted(map(int,capture['cases'])),'source_camera_clock_audit':camera_clock}
        assert max(record['initial_component_maximum_differences'].values())<=1e-7
        checked.append({'iteration':record['iteration'],'case_id':record['case_id'],'media':media,'all_output_hashes_match':True,'cold_start_components_match':True,'qualified':[record['left_independent_pass'],record['right_independent_pass']]})
    result={'status':'PASS','scope':'Artifact/source/recorded initial-state verification, not a replacement for independent physical audit','comparisons':checked,'runs':runs,'verifier_sha256':sha(Path(__file__))}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','comparisons':len(checked),'recording_runs':len(runs)}))


def read_media(path):
    result=subprocess.run(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(path)],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)['streams'][0]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
