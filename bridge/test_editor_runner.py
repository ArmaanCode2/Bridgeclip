"""The editor worker reports fixed failure codes and logs only redacted detail."""

import asyncio
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'engine'))

import editor_runner  # noqa: E402


class RenderingError(Exception):
    pass


class EditorRunnerTests(unittest.TestCase):
    def test_failures_map_to_fixed_codes(self):
        tagged = ValueError('Mark this clip ready before baking it')
        tagged.editor_code = 'not_ready'
        forged = ValueError('x')
        forged.editor_code = '/private/path'
        http = OSError('HTTP 404')
        http.code = 404
        cases = [(tagged, 'not_ready'), (forged, None), (http, None), (RenderingError('FFmpeg failed'), 'render_failed'),
                 (asyncio.CancelledError(), 'cancelled'), (ModuleNotFoundError('cv2'), 'engine_unavailable'),
                 (RuntimeError('boom'), None)]
        for error, code in cases:
            with self.subTest(error=error):
                self.assertEqual(editor_runner.failure_code(error), code)

    def test_failure_log_keeps_the_tail_and_redacts_paths_and_keys(self):
        try:
            try:
                raise RenderingError('FFmpeg failed: banner\n' + 'noise\n' * 20 +
                                     "Error opening /Users/someone/Library/clip.mp4 token=sk-or-secret\nNo such filter: 'perspective'")
            except RenderingError as cause:
                cause.editor_code = 'render_failed'
                raise ValueError('wrapped') from cause
        except ValueError as error:
            with self.assertLogs('editor_runner', 'ERROR') as logs:
                editor_runner.log_failure('export', 'render_failed', error)
        text = '\n'.join(logs.output)
        self.assertIn("No such filter: 'perspective'", text)
        self.assertIn('Editor export failed (render_failed)', text)
        self.assertNotIn('/Users/someone', text)
        self.assertNotIn('sk-or-secret', text)
        self.assertNotIn('banner', text)

    def test_invalid_request_emits_only_a_fixed_result(self):
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([os.path.join(ROOT, 'engine'), os.path.join(ROOT, 'bridge')]),
               'PYTHONDONTWRITEBYTECODE': '1'}
        request = json.dumps({'action': 'delete-everything', 'run': '/Users/someone/private'})
        result = subprocess.run([sys.executable, os.path.join(ROOT, 'bridge', 'editor_runner.py')], input=request,
                                capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout), {'ok': False, 'error': None})
        self.assertIn('Editor request failed (unknown)', result.stderr)
        self.assertNotIn('/Users/someone', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
