"""Tiny synthetic source/guard tests: no network, model, compiler or host install."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from snapdragon_ai_lab import bootstrap_guard as g
from snapdragon_ai_lab import bootstrap_sources as s
from snapdragon_ai_lab.bootstrap_jobs import Runner

class BootstrapSources(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_default_never_opens_lease_or_runs_sources(self):
        with patch.object(s,'unit_controls',side_effect=AssertionError('no execution')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(s.main([]),0)
    def test_fresh_private_canonical_inode_excludes_second_holder(self):
        first=g.acquire_lease(self.root)
        try:
            inode=os.fstat(first).st_ino
            with self.assertRaises(BlockingIOError):g.acquire_lease(self.root)
        finally:os.close(first)
        second=g.acquire_lease(self.root)
        self.assertEqual(os.fstat(second).st_ino,inode);os.close(second)
    def test_legacy_different_inode_is_not_bypassed(self):
        legacy=self.root/'Work/prompt-studio/generation.lock';legacy.parent.mkdir(parents=True);legacy.touch(mode=0o600)
        with self.assertRaisesRegex(g.Stopped,'legacy_lease_inode_mismatch'):g.acquire_lease(self.root)
    def test_anonymous_checkout_uses_direct_https_pin_without_remote(self):
        commands=[];pin='a'*40
        class Stub:
            def call(_,command):
                commands.append(command);args=command[command.index('-C')+2:]
                destination=Path(command[command.index('-C')+1])
                if args[0]=='init':(destination/'.git').mkdir();(destination/'.git/config').write_text('[core]\n bare = false\n')
                if args[:2]==['rev-parse','HEAD']:return pin+'\n'
                return ''
        s.checkout(Stub(),self.root/'checkout','https://github.com/example/source.git',pin)
        flattened=[x for command in commands for x in command]
        self.assertNotIn('remote',flattened);self.assertNotIn('commit',flattened)
        fetch=next(command for command in commands if 'fetch' in command)
        self.assertEqual(fetch[-2:],['https://github.com/example/source.git',pin]);self.assertIn('--depth',fetch);self.assertIn('--no-tags',fetch)
    def test_metadata_and_hardlink_escapes_rejected(self):
        root=self.root/'source';(root/'.git').mkdir(parents=True);config=root/'.git/config';config.write_text('[core]\nworktree=/other\n')
        with self.assertRaisesRegex(g.Stopped,'external_git'):s.audit_git(root)
        config.write_text('[core]\nbare=false\n');item=root/'fixture';item.write_text('tiny');os.link(item,self.root/'shared')
        with self.assertRaisesRegex(g.Stopped,'hardlinks'):s.audit_git(root)
    def test_stub_job_is_bounded_and_requires_no_compiler(self):
        class Okay:
            def require(self):pass
        runner=Runner(Okay(),self.root,self.root,log_cap=1024)
        try:self.assertEqual(runner.call([sys.executable,'-B','-c','print("tiny synthetic job")']).strip(),'tiny synthetic job')
        finally:runner.close()
        self.assertLess((self.root/'commands.log').stat().st_size,1024)
        self.assertEqual(runner.env['GIT_CONFIG_GLOBAL'],'/dev/null');self.assertEqual(runner.env['GIT_TERMINAL_PROMPT'],'0')
        self.assertNotIn('SSH_AUTH_SOCK',runner.env)
    def test_sources_require_100percent_1gib_finite_unit(self):
        original=Path.read_text;cpu=['100000 100000']
        def read(path,*args,**kwargs):
            text=str(path)
            if text=='/proc/self/cgroup':return '0::/user.slice/local-ai-bootstrap-sources.service\n'
            values={'cpu.max':cpu[0],'memory.max':str(g.GIB),'memory.swap.max':'0','pids.max':'64'}
            if text.startswith('/sys/fs/cgroup/'):return values[path.name]
            return original(path,*args,**kwargs)
        result=SimpleNamespace(stdout='RuntimeMaxUSec=15min\nTimeoutStopUSec=10s\nKillMode=control-group\nSendSIGKILL=yes\n')
        with patch.object(Path,'read_text',new=read),patch.object(g.subprocess,'run',return_value=result):
            self.assertEqual(g.unit_controls('sources')[2],900)
            cpu[0]='200000 100000'
            with self.assertRaisesRegex(g.Stopped,'resource_caps'):g.unit_controls('sources')
    def test_real_resource_policy_allows_pageins_stops_outs_and_discharge(self):
        group=self.root/'cgroup';group.mkdir()
        for name in ['memory.swap.current','memory.swap.max']:(group/name).write_text('0')
        guard=g.Guard(group,100);original=Path.read_text;swap=['pswpin 8\npswpout 20\n']
        def read(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return 'MemAvailable: 10000000 kB\n'
            if str(path)=='/proc/vmstat':return swap[0]
            return original(path,*args,**kwargs)
        state={'external_online':True,'battery_percent':80.,'battery_status':'Charging','battery_power_now_uw':-3000000}
        with patch.object(Path,'read_text',new=read),patch.object(g.power,'snapshot',return_value=state):
            guard.poll(starting=True);swap[0]='pswpin 9\npswpout 20\n';guard.poll();self.assertEqual(guard.swap_latest,(9,20))
            swap[0]='pswpin 9\npswpout 21\n'
            with self.assertRaisesRegex(g.Stopped,'global_swapout'):guard.poll()
            swap[0]='pswpin 9\npswpout 20\n';guard.history=[(time.monotonic()-61,{**state,'battery_percent':80.5})]
            with self.assertRaisesRegex(g.Stopped,'sustained_battery_discharge'):guard.poll()
    def test_network_wall_stop_unarms_before_checkpoint(self):
        guard=g.Guard(self.root,100);old=signal.signal(signal.SIGUSR1,guard.signal_stop)
        def poll():
            if guard.network_deadline and time.monotonic()>guard.network_deadline:raise g.Stopped('network_wall_budget')
        guard.poll=poll
        def dribble():time.sleep(2)
        try:
            guard.thread.start();start=time.monotonic()
            with self.assertRaisesRegex(g.Stopped,'network_wall_budget'):guard.network_call(dribble,.05)
            self.assertLess(time.monotonic()-start,1);self.assertFalse(guard.network_active)
        finally:guard.close();signal.signal(signal.SIGUSR1,old)

if __name__=='__main__':unittest.main()
