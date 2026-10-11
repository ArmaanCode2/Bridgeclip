"""Private editor worker. Main validates the library root and owns its lock."""
import asyncio
import json
import logging
import os
import sys
import signal

from bridge_runner import reserve_stdout_for_protocol, emit

logger = logging.getLogger('editor_runner')
ACTIONS = ('review', 'export', 'replace-source', 'scan-cameras')
# Keep in sync with EDITOR_ERROR_CODES in manual_editor.py and editorErrorCodes
# in src/shared/clip-editor.ts. Only these codes cross the bridge.
ERROR_CODES = frozenset({'duration', 'geometry', 'audio', 'invalid', 'project_changed', 'invalid_edit', 'not_ready',
    'source_missing', 'source_incompatible', 'render_failed', 'scan_too_long', 'review_unavailable', 'engine_unavailable',
    'cancelled'})


def failure_code(error):
    """A fixed code for an editor failure, never tool output or private paths."""
    if isinstance(error, asyncio.CancelledError):
        return 'cancelled'
    code = getattr(error, 'editor_code', None)
    if code in ERROR_CODES:
        return code
    if type(error).__name__ == 'RenderingError':
        return 'render_failed'
    if isinstance(error, ImportError):
        return 'engine_unavailable'
    return None


def log_failure(action, code, error):
    """Redacted lines for main's bounded log tail (paths, URLs and keys removed).

    FFmpeg's reason is usually at the end of its output, so keep the last lines
    of each cause. Each line is redacted whole, before any truncation.
    """
    from clip_engine.logging_safety import safe_log_text
    logger.error('Editor %s failed (%s)', action, code or 'unknown')
    seen = set()
    while error is not None and id(error) not in seen and len(seen) < 3:
        seen.add(id(error))
        lines = [line.strip() for line in str(error).splitlines() if line.strip()][-6:] or ['']
        for line in lines:
            logger.error('%s: %s', type(error).__name__, safe_log_text(line)[-300:])
        error = error.__cause__ or error.__context__


def main():
    reserve_stdout_for_protocol()
    action = None
    try:
        from clip_engine.logging_safety import install_safe_logging
        install_safe_logging()
        raw = sys.stdin.read(16385)
        if len(raw) > 16384:
            raise ValueError('Request too large')
        config = json.loads(raw)
        action = config.get('action')
        if action not in ACTIONS or not os.path.isabs(config['run']):
            raise ValueError('Invalid editor action')
        os.environ['LOCAL_MODE'] = 'true'
        from network_guard import install
        install()
        from clip_engine.services.manual_editor import run_editor
        async def work():
            task = asyncio.current_task()
            if os.name != 'nt':
                asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            await run_editor(config, progress=lambda value: emit({'type': 'progress', **value}))
        asyncio.run(work())
        emit({'ok': True})
        return 0
    except (Exception, asyncio.CancelledError) as error:
        code = failure_code(error)
        try:
            log_failure(action if action in ACTIONS else 'request', code, error)
        except Exception:
            pass  # Logging must never hide the protocol result.
        emit({'ok': False, 'error': code})
        return 1


if __name__ == '__main__':
    sys.exit(main())
