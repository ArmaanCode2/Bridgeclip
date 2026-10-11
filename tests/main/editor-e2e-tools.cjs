'use strict'
// Media tools for the clip editor's end-to-end tests. Prefer the staged LGPL
// FFmpeg (engine-bin) and the engine venv, fall back to PATH, and skip with a
// reason when a tool is missing instead of failing on a machine or CI runner
// without them.

const fs = require('node:fs')
const path = require('node:path')
const { execFileSync } = require('node:child_process')

const ROOT = path.resolve(__dirname, '../..')
const exe = (name) => process.platform === 'win32' ? `${name}.exe` : name

function output(command, args, options = {}) {
  try { return execFileSync(command, args, { stdio: ['ignore', 'pipe', 'ignore'], timeout: 30000, ...options }).toString() } catch { return null }
}

function findFfmpeg() {
  const bundled = path.join(ROOT, 'engine-bin', exe('ffmpeg'))
  for (const candidate of [process.env.TEST_FFMPEG, fs.existsSync(bundled) ? bundled : null, 'ffmpeg']) {
    if (candidate && output(candidate, ['-hide_banner', '-version'])) return candidate
  }
  return null
}

/** Encoder args by availability. Chromium plays H.264, so mpeg4 only suits fixtures that are never played. */
function videoEncoder(ffmpeg, { playable = true } = {}) {
  const encoders = output(ffmpeg, ['-hide_banner', '-encoders']) ?? ''
  const has = (name) => new RegExp(`^\\s*V\\S*\\s+${name}\\s`, 'm').test(encoders)
  if (has('libx264')) return ['-c:v', 'libx264', '-preset', 'ultrafast']
  if (has('h264_videotoolbox')) return ['-c:v', 'h264_videotoolbox', '-allow_sw', '1', '-b:v', '4M']
  if (has('libopenh264')) return ['-c:v', 'libopenh264', '-b:v', '4M']
  if (!playable && has('mpeg4')) return ['-c:v', 'mpeg4', '-q:v', '3']
  return null
}

function findEnginePython() {
  const venv = path.join(ROOT, 'engine', '.venv', ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']))
  const env = { ...process.env, PYTHONPATH: path.join(ROOT, 'engine'), PYTHONDONTWRITEBYTECODE: '1' }
  for (const candidate of [process.env.TEST_PYTHON, fs.existsSync(venv) ? venv : null, process.platform === 'win32' ? 'python' : 'python3']) {
    if (candidate && output(candidate, ['-c', 'import clip_engine.services.manual_editor'], { env }) !== null) return candidate
  }
  return null
}

/**
 * FFmpeg, encoder args and (optionally) an engine Python for one test, or null
 * after skipping it. `appEnv` puts a non-bundled FFmpeg first on the app's PATH.
 */
function editorTools(t, { python = false, captions = false, playable = true } = {}) {
  const ffmpeg = findFfmpeg()
  const encoder = ffmpeg ? videoEncoder(ffmpeg, { playable }) : null
  const enginePython = python ? findEnginePython() : null
  const missing = !ffmpeg ? 'FFmpeg (engine-bin/ffmpeg, TEST_FFMPEG or PATH)'
    : !encoder ? 'an H.264 encoder in FFmpeg (libx264, VideoToolbox or OpenH264)'
    : captions && !/^\s*\S+\s+ass\s/m.test(output(ffmpeg, ['-hide_banner', '-filters']) ?? '') ? "FFmpeg's libass `ass` filter"
    : python && !enginePython ? 'a Python with the engine dependencies (engine/.venv, TEST_PYTHON or python3)'
    : null
  if (missing) {
    t.skip(`Needs ${missing}`)
    return null
  }
  const bundled = path.join(ROOT, 'engine-bin', exe('ffmpeg'))
  const appEnv = path.isAbsolute(ffmpeg) && ffmpeg !== bundled ? { PATH: `${path.dirname(ffmpeg)}${path.delimiter}${process.env.PATH ?? ''}` } : {}
  return { ffmpeg, encoder, python: enginePython, appEnv }
}

/** Link the engine folders a development build resolves next to its `out` folder. */
function linkEngine(appDir) {
  for (const dir of ['engine', 'bridge', 'engine-bin']) {
    if (fs.existsSync(path.join(ROOT, dir))) fs.symlinkSync(path.join(ROOT, dir), path.join(appDir, dir), 'junction')
  }
}

module.exports = { editorTools, linkEngine }
