"""shader_split: large chunked KTGL IDRK entries are scanned from a temporary mapping, not joined in RAM."""
import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/gfx/tools'))
_had = os.environ.get('DM_GAME'); os.environ['DM_GAME'] = '0400A3A01EE4C000'
import shader_split as ss
if _had is None: del os.environ['DM_GAME']
else: os.environ['DM_GAME'] = _had


def glslc_output(sph_type, marker=0):
    code = bytearray(ss.GFX_MAGIC + bytes(0x2c))
    code += struct.pack('<I', sph_type) + bytes(0x4c)
    code += bytes.fromhex('0f00000000000000') * 8
    code[-1] = marker
    control = bytearray(0x880)
    control[:4] = ss.CTL_MAGIC
    struct.pack_into('<I', control, 0x20, len(control))
    struct.pack_into('<I', control, 0x704, len(code))
    output = bytearray(0x50) + control + code
    output[:4] = bytes.fromhex('91688619')
    struct.pack_into('<I', output, 0x44, len(output))
    return bytes(output)


def idrk_file(data):
    chunks = []
    for offset in range(0, len(data), 0x4000):
        compressed = zlib.compress(data[offset:offset + 0x4000])
        chunks.append(struct.pack('<H', len(compressed)) + bytes(8) + compressed)
    stored = b''.join(chunks) + bytes(10)
    header = ss.IDRK + struct.pack('<QQQIIII', 0x30 + len(stored), len(stored), len(data), 0, 0, 0, 0x400000)
    return header + stored


class IrdkStreaming(unittest.TestCase):
    def test_selected_glslc_keeps_metadata_for_every_stage(self):
        payload = glslc_output(0x0461) + glslc_output(0x1462)
        all_shaders = ss.glslc_shaders(payload, '/selected', need=False)
        keys = {ss.skey(ctl, code): (where, stage) for where, stage, ctl, code in all_shaders}
        selected_key = next(key for key, (_, stage) in keys.items() if stage == 'vs')
        metadata = {}

        selected = ss.glslc_shaders(payload, '/selected', need=False, selected={selected_key}, metadata=metadata)

        self.assertEqual({ss.skey(ctl, code) for _, _, ctl, code in selected}, {selected_key})
        self.assertEqual(metadata, keys)

    def test_selected_idrk_keeps_full_metadata_and_selected_bytes(self):
        data = b'_S2G3000' + glslc_output(0x0461) + glslc_output(0x1462)
        shaders = ss.glslc_shaders(data, '/container', need=False)
        keys = {ss.skey(ctl, code): (where, stage) for where, stage, ctl, code in shaders}
        selected_key = next(key for key, (_, stage) in keys.items() if stage == 'vs')
        found, metadata = [], {}

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / 'sample.fdata'
            path.write_bytes(idrk_file(data))
            ss.idrk_shaders(str(path), '/sample.fdata', found, selected={selected_key}, metadata=metadata)

        self.assertEqual({ss.skey(ctl, code) for _, _, ctl, code in found}, {selected_key})
        self.assertEqual({key: stage for key, (_, stage) in metadata.items()}, {key: stage for key, (_, stage) in keys.items()})
        self.assertTrue(all(where.startswith('/sample.fdata#IDRK@0x30') for where, _ in metadata.values()))

    def test_selected_scan_keeps_map_keys_but_only_selected_payloads(self):
        data = b'_S2G3000' + glslc_output(0x0461) + glslc_output(0x1462)
        shaders = ss.glslc_shaders(data, '/container', need=False)
        selected_key = next(ss.skey(ctl, code) for _, stage, ctl, code in shaders if stage == 'vs')

        with tempfile.TemporaryDirectory() as work, mock.patch.object(ss, 'SPLIT_SELECTED', {selected_key}):
            path = Path(work) / 'sample.fdata'
            path.write_bytes(idrk_file(data))
            relative, uniq, _pairs, extra = ss.scan_file((str(path), 'sample.fdata'))

        self.assertEqual(relative, 'sample.fdata')
        self.assertEqual(set(uniq), {ss.skey(ctl, code) for _, _, ctl, code in shaders})
        self.assertIsNotNone(uniq[selected_key][2])
        self.assertEqual([key for key, value in uniq.items() if value[2] is None], [next(key for key in uniq if key != selected_key)])
        self.assertEqual(extra, {})

    def test_selected_split_keeps_complete_map_and_compacts_package(self):
        data = b'_S2G3000' + glslc_output(0x0461, 1) + glslc_output(0x0461, 2) + glslc_output(0x1462)
        shaders = ss.glslc_shaders(data, '/container', need=False)
        selected_key = ss.skey(shaders[1][2], shaders[1][3])

        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            romfs = root / 'romfs'
            assets = romfs / 'motor_asset_re' / 'KIDS_PLATFORM_OUNCE'
            assets.mkdir(parents=True)
            (assets / 'sample.fdata').write_bytes(idrk_file(data))
            main = root / 'main'
            main.write_bytes(b'NSO0' + bytes(0xfc))

            def split(name, selected):
                pkg = root / name / 'pkg'
                pkg.mkdir(parents=True)
                outputs = [pkg, root / name / 'map.tsv', root / name / 'facts.tsv', root / name / 'pairs.tsv', root / name / 'pso.tsv']
                argv = ['shader_split.py', str(romfs), str(main), *(str(path) for path in outputs)]
                with mock.patch.object(ss, 'SPLIT_SELECTED', selected), mock.patch.object(sys, 'argv', argv), \
                     mock.patch.dict(os.environ, {'J': '1', 'DM_SPLIT_RESUME': ''}), contextlib.redirect_stdout(io.StringIO()):
                    ss.main()
                records = sum(path.read_bytes().count(b'BTCDBTCD') for path in pkg.glob('*.bin'))
                return outputs[1].read_bytes(), records

            full_map, full_records = split('full', None)
            selected_map, selected_records = split('selected', {selected_key})

        self.assertEqual(selected_map, full_map)
        self.assertEqual(full_records, 3)
        self.assertEqual(selected_records, 1)

    def test_scans_full_entry_from_mmap(self):
        data = b'_S2G3000' + bytes(range(256)) * 512
        seen = []
        scan = ss.glslc_shaders
        tempfile_kwargs = []
        temporary_file = ss.tempfile.TemporaryFile

        def capture_temporary_file(*args, **kwargs):
            tempfile_kwargs.append(kwargs)
            return temporary_file(*args, **kwargs)

        def capture(decoded, label, need=True, skip=(), **kwargs):
            self.assertNotIsInstance(decoded, bytes)
            self.assertEqual(bytes(decoded), data)
            seen.append((label, need))
            return scan(decoded, label, need=need, skip=skip, **kwargs)

        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / '0x38f972af.fdata'
            path.write_bytes(idrk_file(data))
            with mock.patch.dict(os.environ, NX_PACK_WORK=work), mock.patch.object(ss.tempfile, 'TemporaryFile', side_effect=capture_temporary_file), mock.patch.object(ss, 'glslc_shaders', side_effect=capture):
                ss.idrk_shaders(str(path), '/0x38f972af.fdata', [])

        self.assertEqual(seen, [('/0x38f972af.fdata#IDRK@0x30', False)])
        self.assertEqual(tempfile_kwargs, [{}])


if __name__ == '__main__':
    unittest.main()