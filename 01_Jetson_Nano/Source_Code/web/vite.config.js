
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// ======================================================
// VISION SWINE V2 - Vite Configuration
// ======================================================

export default defineConfig({
  plugins: [react()],

  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,

    // ==================================================
    // API Proxy - Jetson Nano and NAS
    // ==================================================
    proxy: {
      '/jetson-api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/jetson-api/, ''),
      },

      '/nas-api': {
        target: 'http://100.123.171.62:8002',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/nas-api/, ''),
      },
    },
  },

  preview: {
    host: '0.0.0.0',
    port: 4173,
  },
})
