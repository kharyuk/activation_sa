import subprocess

sh_script_templates = {
    'base': '../templates/base.sh',
    'slurm': '../templates/slurm.sh',
}

def build_sh_exp_files(
    config_dict,
    script_path,
    experiment_file_path,
    which='base',
    slurm_config_dict=None,
    slurm_path_variables_dict=None
):
    assert which in sh_script_templates.keys()
    if which == 'slurm':
        assert slurm_config_dict is not None
        sbatch_lines = [
            f'#SBATCH --{key}={value}\n' for key, value in slurm_config_dict.items()
        ]
        assert slurm_path_variables_dict is not None
        path_lines = [
            f'{key}="{value}"\n' for key, value in slurm_path_variables_dict.items()
        ]
    with open(sh_script_templates[which], 'r') as f:
        sh_content = f.readlines()
        
    config_string = ' '.join(f'--{key} {value}' for key, value in config_dict.items())
    run_line = f'python {experiment_file_path}'
    run_line = f'{run_line} {config_string}'
    
    if which == 'slurm':
        sh_content = sh_content[:1] + sbatch_lines + ['\n'] + path_lines + sh_content[1:]
        assert slurm_config_dict['nodes'] == 1,  'not supported: multiple-nodes'
        run_line1 = (
            f"srun -N 1 -n 1 -c "
            f"{slurm_config_dict['cpus-per-task']} "
            f"{run_line} --do_compute_activations 1 --do_compute_values 0 \n"
            #f"--do_use_slurm 1\n"
        )
        run_line2 = (
            f"srun -N 1 -n 1 "#{slurm_config_dict['ntasks']} "
            f"-c {slurm_config_dict['cpus-per-task']} "
            f"{run_line} --do_compute_activations 0 --do_compute_values 1 \n"
            #f"--do_use_slurm 1\n"
        )
        sh_content = sh_content+['\n', run_line1, run_line2]
    else:
        sh_content = sh_content+['\n', run_line]
    
    with open(script_path, 'w') as f:
        f.writelines(sh_content)
    #! chmod +x {current_script_filename}
    subprocess.run(
        f'chmod +x {script_path}',
        shell=True,
        capture_output=False,
        check=False
    )
    

# okke, we may use <<re>>, but...

def convert_dict2argstr(a_dict):
    #rv = ', '.join(f'("{key}", "{value}", "{type(value).__name__}")' for key, value in a_dict.items())
    rv = ''
    for key, value in a_dict.items():
        if len(rv) > 0:
            rv += ', '
        vtype = type(value).__name__
        if vtype in ('tuple', 'list'):
            vtype_inner = ', '.join(f'{type(iv).__name__}' for iv in value)
            vtype = f'{vtype}[{vtype_inner}]'
        current = f'("{key}", "{value}", "{vtype}")'        
        rv += f'{current}'
    return f"'({rv})'"


def convert_argstr2dict(arg_str):
    rv = {}
    tmp = arg_str.replace("'", '')
    tmp = tmp[1:-1]
    tmp = tmp.split('), ')
    for t in tmp:
        if not t.endswith(')'): t += ')'
        key, value, value_type = eval(t)
        if value_type.startswith('tuple') or value_type.startswith('list'): # ..iterable?..
            value_type, llv_types = value_type.split('[')
            value = [eval(llv_type)(val) for (val, llv_type) in zip(eval(value), llv_types[:-1].split(', '))]
        value = eval(value_type)(value)
        rv[key] = value
    return rv

def convert_list2argstr(a_list):
    rv = ', '.join(f'"{value}"' for value in a_list)
    return f"'[{rv}]'"
