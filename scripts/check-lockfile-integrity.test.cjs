'use strict'
const test = require('node:test')
const assert = require('node:assert/strict')
const { readFileSync } = require('fs')
const { join } = require('path')
const { findLockfileProblems } = require('./check-lockfile-integrity.cjs')

const good = {
  resolved: 'https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz',
  integrity: 'sha512-ZP5s6zW+Kz2jX1xkSsyq6NQ4h2S5tI7gImuFsE9UQ6JR1Pz8x7v9G0kpMbhZ4uYYszHkzSrUHtWCsM0M0NpXDQ=='
}

test('accepts an entry with a registry URL and sha512 integrity', () => {
  assert.deepEqual(findLockfileProblems({ packages: { '': {}, 'node_modules/left-pad': { version: '1.3.0', ...good } } }), [])
})

test('ignores the root project and workspace links', () => {
  assert.deepEqual(findLockfileProblems({ packages: { '': { name: 'app' }, 'node_modules/ws': { resolved: 'packages/ws', link: true } } }), [])
})

test('reports entries that only pin a version', () => {
  const problems = findLockfileProblems({ packages: { 'node_modules/react': { version: '19.2.4', dev: true } } })
  assert.equal(problems.length, 1)
  assert.match(problems[0], /node_modules\/react: missing resolved\/integrity/)
})

test('reports non-registry sources and weak hashes', () => {
  const problems = findLockfileProblems({ packages: {
    'node_modules/a': { ...good, resolved: 'https://example.com/a.tgz' },
    'node_modules/b': { ...good, integrity: 'sha1-2jmj7l5rSw0yVb/vlWAYkK/YBwk=' }
  } })
  assert.deepEqual(problems.map((p) => p.split(':')[0]), ['node_modules/a', 'node_modules/b'])
})

test('reports registry tarballs for a different package, version or alias', () => {
  const problems = findLockfileProblems({ packages: {
    'node_modules/react': { version: '1.3.0', ...good },
    'node_modules/left-pad': { version: '1.2.0', ...good },
    'node_modules/pad': { version: '1.3.0', name: 'left-pad', ...good },
    'node_modules/@scope/left-pad': { version: '1.3.0', ...good },
    'node_modules/a/node_modules/@scope/b': { version: '2.0.0', ...good, resolved: 'https://registry.npmjs.org/@scope/b/-/b-2.0.0.tgz' },
    'node_modules/c': { version: '1.0.0', ...good, resolved: 'https://registry.npmjs.org/c/-/c-1.0.0.tgz' }
  } })
  assert.deepEqual(problems.map((p) => p.split(':')[0]), ['node_modules/react', 'node_modules/left-pad', 'node_modules/pad', 'node_modules/@scope/left-pad'])
  assert.match(problems[0], /not the registry tarball for this package and version/)
})

test('the committed lockfile passes', () => {
  const lock = JSON.parse(readFileSync(join(__dirname, '..', 'package-lock.json'), 'utf8'))
  assert.deepEqual(findLockfileProblems(lock), [])
})
