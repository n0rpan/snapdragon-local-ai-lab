import json
from pathlib import Path
from importlib import resources
import importlib.util
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.plugin_install import install,PLUGIN_ID
from snapdragon_ai_lab.platform_support import detect

# The home directory of whoever runs the tests must never leak into generated files.
DEVELOPER_HOME=__import__('pwd').getpwuid(__import__('os').getuid()).pw_dir

class PluginInstallation(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'omarchy';self.root.mkdir()
        (self.root/'shell.json').write_text('{"bar":{}}\n')
        self.cli=self.root/'local-ai';self.cli.write_text('#!/bin/sh\nexit 0\n');self.cli.chmod(0o700)
        self.info={'omarchy_command_present':True,'snapdragon_x_detected':True}

    def call(self,**kwargs):
        return install(config_root=self.root,cli_path=self.cli,platform_info=self.info,**kwargs)

    def test_dry_run_writes_nothing(self):
        before=set(self.root.iterdir());result=self.call(dry_run=True)
        self.assertTrue(result['dry_run'])
        self.assertEqual(before,set(self.root.iterdir()))

    def test_unsupported_hardware_requires_explicit_override(self):
        self.info['snapdragon_x_detected']=False
        with self.assertRaises(RuntimeError):self.call(dry_run=True)
        self.assertFalse(self.call(dry_run=True,allow_unsupported=True)['snapdragon_x_detected'])

    def test_failed_layout_change_restores_original_plugin_and_config(self):
        target=self.root/'plugins'/PLUGIN_ID;target.mkdir(parents=True);(target/'owner.txt').write_text('original')
        original=(self.root/'shell.json').read_bytes()
        def fail(command,**kwargs):
            (self.root/'shell.json').write_text('partial change')
            raise subprocess.CalledProcessError(1,command)
        with self.assertRaises(subprocess.CalledProcessError):self.call(runner=fail)
        self.assertEqual((target/'owner.txt').read_text(),'original')
        self.assertEqual((self.root/'shell.json').read_bytes(),original)

    def test_assets_use_portable_home_path_and_right_placement(self):
        calls=[];result=self.call(runner=lambda command,**kwargs:calls.append(command))
        target=self.root/'plugins'/PLUGIN_ID
        manifest=json.loads((target/'manifest.json').read_text())
        self.assertEqual(manifest['id'],PLUGIN_ID)
        panel=(target/'Panel.qml').read_text()
        self.assertNotIn(DEVELOPER_HOME,panel)
        self.assertIn('Quickshell.env("HOME")',panel)
        self.assertIn('root.controlExecutable',panel)
        self.assertEqual(calls,[['omarchy','bar','put',PLUGIN_ID,'--section','right']])
        self.assertTrue(Path(result['backup_directory'],'shell.json').exists())

    def test_installs_usable_packaged_planning_helper_beside_regular_cli(self):
        self.call(runner=lambda *a,**k:None)
        helper=self.cli.with_name('launch_local_plan.py')
        expected=resources.files('snapdragon_ai_lab').joinpath('runtime/launch_local_plan.py').read_bytes()
        self.assertEqual(helper.read_bytes(),expected)
        self.assertEqual(helper.stat().st_mode & 0o777,0o700)
        spec=importlib.util.spec_from_file_location('installed_plan_helper',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        overlay=json.loads(module.environment_for_named_agent({})['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(overlay['default_agent'],'local-plan')
        self.assertEqual(overlay['policies'][-2:],module.READ_ONLY_POLICIES)
        self.assertEqual(self.cli.read_text(),'#!/bin/sh\nexit 0\n')

    def test_failed_install_restores_or_removes_planning_helper(self):
        helper=self.cli.with_name('launch_local_plan.py')
        for existed in (False,True):
            with self.subTest(existed=existed):
                helper.unlink(missing_ok=True)
                if existed:helper.write_bytes(b'previous helper');helper.chmod(0o640)
                with self.assertRaises(subprocess.CalledProcessError):
                    self.call(runner=lambda *a,**k:(_ for _ in ()).throw(subprocess.CalledProcessError(1,a[0])))
                self.assertEqual(helper.exists(),existed)
                if existed:
                    self.assertEqual(helper.read_bytes(),b'previous helper')
                    self.assertEqual(helper.stat().st_mode & 0o777,0o640)

    def test_symlinked_control_or_helper_is_rejected_before_writes(self):
        target=self.root/'unrelated';target.write_text('preserve')
        helper=self.cli.with_name('launch_local_plan.py');helper.symlink_to(target)
        before=self.tree()
        with self.assertRaises(RuntimeError):self.call()
        self.assertEqual(self.tree(),before)
        helper.unlink();self.cli.unlink();self.cli.symlink_to(target)
        before=self.tree()
        with self.assertRaises(RuntimeError):self.call()
        self.assertEqual(self.tree(),before)

    def test_soc_detection_uses_compatible_not_npu_permissions(self):
        compatible=self.root/'compatible';compatible.write_bytes(b'lenovo,example\0qcom,x1e80100\0')
        self.assertTrue(detect('aarch64',compatible)['snapdragon_x_detected'])
        self.assertFalse(detect('x86_64',compatible)['snapdragon_x_detected'])

    def migration_fixture(self,section='right'):
        layout={'left':[{'id':'omarchy.workspaces'}],
                'right':[{'id':'omarchy.tray'},{'id':'prototype.local-ai','custom':{'label':'offline'}},{'id':'omarchy.audio'}],
                'tray':[]}
        if section=='tray':layout['tray'].append(layout['right'].pop(1))
        document={'bar':{'layout':layout,'height':33},'idle':{'lock':600}}
        (self.root/'shell.json').write_text(json.dumps(document))
        for name,value in [('prototype.local-ai','prototype'),(PLUGIN_ID,'previous public')]:
            directory=self.root/'plugins'/name;directory.mkdir(parents=True)
            (directory/'owner.txt').write_text(value)
        return document

    def test_migration_preserves_position_settings_and_archives_both_plugins(self):
        original=self.migration_fixture()
        result=self.call(replace_plugin='prototype.local-ai',runner=lambda *a,**k:self.fail('migration must not invoke bar put'))
        expected=original;expected['bar']['layout']['right'][1]['id']=PLUGIN_ID
        self.assertEqual(json.loads((self.root/'shell.json').read_text()),expected)
        self.assertFalse((self.root/'plugins/prototype.local-ai').exists())
        backup=Path(result['backup_directory'])
        self.assertEqual((backup/'replaced-plugin/owner.txt').read_text(),'prototype')
        self.assertEqual((backup/'plugin/owner.txt').read_text(),'previous public')
        self.assertEqual(json.loads((backup/'shell.json').read_text())['bar']['layout']['right'][1]['id'],'prototype.local-ai')

    def test_tray_migration_becomes_visible_at_right_index_one(self):
        self.migration_fixture('tray');self.call(replace_plugin='prototype.local-ai')
        layout=json.loads((self.root/'shell.json').read_text())['bar']['layout']
        self.assertEqual(layout['tray'],[])
        self.assertEqual([x['id'] for x in layout['right']],['omarchy.tray',PLUGIN_ID,'omarchy.audio'])

    def test_migration_failure_restores_shell_and_both_directories(self):
        self.migration_fixture();original=(self.root/'shell.json').read_bytes()
        def fail(path,data):
            path.write_bytes(b'partial update')
            raise OSError('synthetic layout failure')
        with patch('snapdragon_ai_lab.plugin_install.write_layout',side_effect=fail):
            with self.assertRaises(OSError):self.call(replace_plugin='prototype.local-ai')
        self.assertEqual((self.root/'shell.json').read_bytes(),original)
        self.assertEqual((self.root/'plugins/prototype.local-ai/owner.txt').read_text(),'prototype')
        self.assertEqual((self.root/'plugins'/PLUGIN_ID/'owner.txt').read_text(),'previous public')

    def tree(self):
        return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}

    def test_migration_dry_run_and_invalid_ids_write_nothing(self):
        self.migration_fixture();before=self.tree()
        self.call(replace_plugin='prototype.local-ai',dry_run=True)
        self.assertEqual(self.tree(),before)
        for value in ['../escape','prototype/local-ai',PLUGIN_ID,'a..b','a.b;touch x','']:
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):self.call(replace_plugin=value)
                self.assertEqual(self.tree(),before)
        self.assertFalse((self.root/'plugin-backups').exists())

    def test_missing_duplicate_or_public_layout_ids_refused_without_writes(self):
        for kind in ['missing','duplicate','public','unrelated-duplicate']:
            with self.subTest(kind=kind):
                # Reuse directories, reset only the synthetic layout.
                layout={'right':[{'id':'prototype.local-ai'}]}
                old=self.root/'plugins/prototype.local-ai';old.mkdir(parents=True,exist_ok=True)
                if kind=='missing':layout['right']=[]
                if kind=='duplicate':layout['tray']=[{'id':'prototype.local-ai'}]
                if kind=='public':layout['tray']=[{'id':PLUGIN_ID}]
                if kind=='unrelated-duplicate':layout['tray']=[{'id':'omarchy.audio'},{'id':'omarchy.audio'}]
                (self.root/'shell.json').write_text(json.dumps({'bar':{'layout':layout}}))
                before=self.tree()
                with self.assertRaises(RuntimeError):self.call(replace_plugin='prototype.local-ai')
                self.assertEqual(self.tree(),before)
                self.assertFalse((self.root/'plugin-backups').exists())

    def test_symlinked_root_is_rejected(self):
        link=self.root.parent/'linked';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(RuntimeError):
            install(config_root=link,cli_path=self.cli,platform_info=self.info,dry_run=True)

    def test_migration_staging_failure_leaves_both_plugins_unchanged(self):
        self.migration_fixture();before=self.tree()
        with patch('snapdragon_ai_lab.plugin_install.copy_assets',side_effect=OSError('synthetic staging failure')):
            with self.assertRaises(OSError):self.call(replace_plugin='prototype.local-ai')
        after={key:value for key,value in self.tree().items() if not key.startswith('plugin-backups/')}
        self.assertEqual(after,before)

    def test_symlinked_plugin_directory_is_rejected_before_backup(self):
        real=self.root/'real-plugins';real.mkdir()
        (self.root/'plugins').symlink_to(real,target_is_directory=True)
        with self.assertRaises(RuntimeError):self.call(dry_run=True)
        self.assertFalse((self.root/'plugin-backups').exists())

if __name__=='__main__':unittest.main()
