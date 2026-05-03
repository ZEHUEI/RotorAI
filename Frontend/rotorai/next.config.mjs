/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        // Strict headers only for 3DGS viewer
        source: '/gaussian(.*)',
        headers: [
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
        ],
      },
      {
        // Everything else — relaxed
        source: '/((?!gaussian).*)',
        headers: [
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'unsafe-none',
          },
          {
            key: 'Content-Security-Policy',
            value:
              "frame-src 'self' https://www.youtube-nocookie.com https://www.youtube.com;",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
