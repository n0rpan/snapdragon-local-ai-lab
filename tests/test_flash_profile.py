import copy
import unittest
from snapdragon_ai_lab.flash_profile import summarize, cpu_counters, cpu_utilization, process_identity, summarize_io_trace, classify_cpu_symbol
from pathlib import Path
import tempfile


class Attribution(unittest.TestCase):
    def test_leaf_dot_cannot_be_called_dequantization_only(self):
        self.assertEqual(classify_cpu_symbol('ggml_vec_dot_iq3_xxs_q8_K'), 'quantized_dot_including_unpack')
        self.assertEqual(classify_cpu_symbol('dequantize_row_iq3_xxs'), 'standalone_dequantization')
        self.assertEqual(classify_cpu_symbol('[unknown]'), 'other_or_unresolved')

    def fixture(self):
        return {'timings': {'total_wall_seconds': 12}, 'native_requests': [
            {'tokens': 4, 'tok_s': 2, 'prefill_s': 3, 'flash_wait_s_tok': .1,
             'compute_s_tok': .3, 'mgmt_s_tok': .1, 'io_s_tok': 1, 'read_mib': 8},
            {'tokens': 6, 'tok_s': 1, 'prefill_s': 0, 'flash_wait_s_tok': .2,
             'compute_s_tok': .7, 'mgmt_s_tok': .1, 'io_s_tok': 2, 'read_mib': 12}]}

    def test_weighted_accounting_and_nonadditive_lanes(self):
        source=self.fixture(); before=copy.deepcopy(source); out=summarize(source)
        self.assertEqual(out['aggregate_decode_tokens_per_second'], 1.25)
        self.assertEqual(out['other_wall_seconds'], 1)
        self.assertEqual(out['decode_io_lane_busy_seconds_nonadditive'], 16)
        self.assertEqual(out['decode_bytes_per_token'], 2*1024**2)
        self.assertFalse(out['attribution_complete'])
        self.assertIsNone(out['direct_decode_compute_seconds'])
        self.assertEqual(source, before)

    def test_refuse_bad_numbers_and_inconsistent_time(self):
        for key,value in [('tok_s',0),('tokens',True),('read_mib',float('nan')),
                          ('compute_s_tok',-.1),('compute_s_tok',8)]:
            source=self.fixture();source['native_requests'][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):summarize(source)
        source=self.fixture();source['timings']['total_wall_seconds']=1
        with self.assertRaises(ValueError):summarize(source)

    def test_do_not_export_unknown_input_fields(self):
        source=self.fixture();source['auth']='private';source['native_requests'][0]['text']='private'
        self.assertNotIn('private',str(summarize(source)))

    def test_guest_is_not_double_counted_and_iowait_is_not_busy(self):
        before=cpu_counters('cpu 1 2 3 4\ncpu0 10 0 0 20 5 0 0 0 8 0\n')
        after=cpu_counters('cpu0 20 0 0 25 10 0 0 0 16 0\n')
        self.assertEqual(before, {'cpu0': (35,25)})
        self.assertEqual(cpu_utilization(before,after), {'cpu0':50})
        self.assertIsNone(cpu_utilization(before,before)['cpu0'])
        with self.assertRaises(ValueError):cpu_utilization(after,before)

    def test_identity_handles_spaces_and_parentheses_in_comm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'12').mkdir()
            fields=['0']*20;fields[0]='S';fields[19]='9876'
            (root/'12/stat').write_text('12 (name with ) parenthesis) '+' '.join(fields))
            self.assertEqual(process_identity(12,root),9876)
            with self.assertRaises(ValueError):process_identity(True,root)

    def test_per_layer_trace_does_not_confuse_hits_with_padded_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.csv'
            p.write_text('# cache,1,1,12,3,10,8\n1,1,12,3,9,0,2,0,0,100,4096,9000\n# frame,1,1,12,100,200\n')
            out=summarize_io_trace(p);item=out['layers'][0]
            self.assertEqual(item['hit_percent'],80)
            self.assertEqual(item['requested_bytes'],100)
            self.assertEqual(item['read_bytes'],4096)
            self.assertFalse(out['lane_busy_time_is_wall_additive'])
            self.assertEqual(out['frames'][0]['end_ns'],200)
            with self.assertRaises(ValueError):summarize_io_trace(p,max_bytes=1)
            p.write_text('# cache,1,1,12,3,2,3\n')
            with self.assertRaises(ValueError):summarize_io_trace(p)
