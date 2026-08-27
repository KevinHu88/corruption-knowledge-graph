import { defineConfig, globalIgnores } from 'eslint/config'
import nextVitals from 'eslint-config-next/core-web-vitals'
import nextTypeScript from 'eslint-config-next/typescript'

export default defineConfig([
  ...nextVitals,
  ...nextTypeScript,
  globalIgnores([
    '.next/**',
    'out/**',
    'build/**',
    'next-env.d.ts',
    // Upstream components target APIs that are not part of this project yet.
    // Keep them as reference code, but lint only the integrated application.
    'src/components/**',
    'src/store/**',
    'src/types/**',
    'src/lib/api.ts',
    'jest.config.js',
    'next.config.js',
    'postcss.config.js',
    'tailwind.config.js',
  ]),
])
