"""Install the optional Omarchy panel after explicit platform preflight."""
from datetime import datetime, timezone
from importlib import resources
import os
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

from .platform_support import detect

PLUGIN_ID='snapdragon.local-ai'

def migration_layout(shell,old):
    """Validate before touching files; retain widget settings and unrelated layout."""
    try:
        document=json.loads(shell)
        layout=document['bar']['layout']
        if not isinstance(layout,dict):raise ValueError()
        matches=[];seen=set()
        for section,entries in layout.items():
            if not isinstance(entries,list):raise ValueError()
            for index,entry in enumerate(entries):
                if not isinstance(entry,dict) or not isinstance(entry.get('id'),str):raise ValueError()
                if entry['id'] in seen:raise RuntimeError('Duplicate layout IDs must be resolved before migration')
                seen.add(entry['id'])
                if entry['id']==PLUGIN_ID:raise RuntimeError('Public plugin already occurs in layout; refusing duplicates')
                if entry['id']==old:matches.append((section,index))
        if len(matches)!=1:raise RuntimeError('Replacement plugin must occur exactly once in bar.layout')
        section,index=matches[0]
        if section not in ('left','centerBegin','center','centerEnd','tray','right'):raise ValueError()
        entry=layout[section].pop(index)
        entry['id']=PLUGIN_ID
        if section=='right':layout[section].insert(index,entry)
        else:
            right=layout.setdefault('right',[])
            # After the tray toggle, or first when there is no toggle.
            position=next((i+1 for i,item in enumerate(right) if item['id']=='omarchy.tray'),0)
            right.insert(position,entry)
        return (json.dumps(document,indent=2)+'\n').encode()
    except (ValueError,KeyError,TypeError) as error:
        raise RuntimeError('Migration requires a valid explicit bar.layout') from error

def write_layout(path,data):
    """Atomic hot-reload update; kept separate for rollback failure tests."""
    descriptor,name=tempfile.mkstemp(prefix='.local-ai-layout-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:stream.write(data)
        shutil.copymode(path,name)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def copy_assets(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for entry in source.iterdir():
        target=destination/entry.name
        if entry.is_dir():copy_assets(entry,target)
        else:target.write_bytes(entry.read_bytes())

def install(*,dry_run=False,allow_unsupported=False,replace_plugin=None,config_root=None,cli_path=None,platform_info=None,runner=subprocess.run):
    info=detect() if platform_info is None else platform_info
    root=Path.home()/'.config/omarchy' if config_root is None else Path(config_root)
    cli=Path.home()/'.local/bin/local-ai' if cli_path is None else Path(cli_path)
    if replace_plugin is not None and (not isinstance(replace_plugin,str) or
            not re.fullmatch(r'[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+',replace_plugin) or
            len(replace_plugin)>128 or replace_plugin==PLUGIN_ID):
        raise RuntimeError('Replacement must be a distinct safe dotted plugin ID')
    for directory in (root,root/'plugins',root/'plugin-backups'):
        if directory.is_symlink():raise RuntimeError('Refusing a symlinked installation root')
    if (root/'shell.json').is_symlink():raise RuntimeError('Refusing a symlinked shell.json')
    if not info.get('omarchy_command_present'):raise RuntimeError('Omarchy is required for this optional panel')
    if not info.get('snapdragon_x_detected') and not allow_unsupported:
        raise RuntimeError('Snapdragon X was not detected; contributors may explicitly use --allow-unsupported')
    if not (root/'shell.json').is_file():raise RuntimeError('Existing Omarchy shell.json was not found')
    if cli.is_symlink() or not cli.is_file() or not os.access(cli,os.X_OK):
        raise RuntimeError('Install the local-ai control command first; this preview does not install the model runtime')
    helper=cli.with_name('launch_local_plan.py')
    if helper.is_symlink() or (helper.exists() and not helper.is_file()):
        raise RuntimeError('Planning helper must be a regular file beside local-ai')
    target=root/'plugins'/PLUGIN_ID
    if target.is_symlink():raise RuntimeError('Refusing to replace a symlinked plugin directory')
    if target.exists() and not target.is_dir():raise RuntimeError('Plugin target must be a directory')
    old=root/'plugins'/replace_plugin if replace_plugin else None
    replacement=None
    if old is not None:
        if old.is_symlink() or not old.is_dir():raise RuntimeError('Replacement must be an existing non-symlink user plugin directory')
        replacement=migration_layout((root/'shell.json').read_bytes(),replace_plugin)
    result={'plugin_id':PLUGIN_ID,'section':'right','dry_run':dry_run,
            'snapdragon_x_detected':bool(info.get('snapdragon_x_detected')),
            'replaces_existing_plugin':target.exists(),'starts_models':False,
            'planning_helper':str(helper)}
    if old is not None:result['replaced_plugin_id']=replace_plugin
    if dry_run:return result
    backups=root/'plugin-backups';backups.mkdir(parents=True,exist_ok=True)
    backup=Path(tempfile.mkdtemp(prefix=PLUGIN_ID+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-',dir=backups))
    shutil.copy2(root/'shell.json',backup/'shell.json')
    helper_existed=helper.exists()
    if helper_existed:shutil.copy2(helper,backup/'launch_local_plan.py')
    moved=False;installed=False;old_moved=False;helper_installed=False
    try:
        with tempfile.TemporaryDirectory(prefix='.local-ai-plugin-stage-',dir=root) as directory:
            staged=Path(directory)/PLUGIN_ID
            copy_assets(resources.files('snapdragon_ai_lab').joinpath('omarchy_plugin'),staged)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():shutil.move(str(target),str(backup/'plugin'));moved=True
            staged.replace(target);installed=True
        # local-ai resolves this sibling at launch. Install the packaged helper
        # and restore the previous helper if panel setup fails.
        descriptor,name=tempfile.mkstemp(prefix='.local-ai-plan-',dir=cli.parent)
        try:
            with os.fdopen(descriptor,'wb') as stream:
                stream.write(resources.files('snapdragon_ai_lab').joinpath('runtime/launch_local_plan.py').read_bytes())
                os.fchmod(stream.fileno(),0o700)
            os.replace(name,helper);helper_installed=True
        finally:
            if os.path.exists(name):os.unlink(name)
        if old is None:
            runner(['omarchy','bar','put',PLUGIN_ID,'--section','right'],check=True,capture_output=True,text=True)
        else:
            shutil.move(str(old),str(backup/'replaced-plugin'));old_moved=True
            write_layout(root/'shell.json',replacement)
    except Exception:
        if helper_installed:
            if helper_existed:shutil.copy2(backup/'launch_local_plan.py',helper)
            else:helper.unlink(missing_ok=True)
        if installed and target.exists():shutil.rmtree(target)
        if moved:shutil.move(str(backup/'plugin'),str(target))
        if old_moved:shutil.move(str(backup/'replaced-plugin'),str(old))
        shutil.copy2(backup/'shell.json',root/'shell.json')
        raise
    result['backup_directory']=str(backup)
    return result
