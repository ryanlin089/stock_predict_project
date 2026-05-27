import type { NextConfig } from "next";

function normalizeApiProxyBaseUrl(value: string | undefined) {
  if (!value) {
    return undefined;
  }

  const trimmedValue = value.replace(/\/$/, "");
  if (/^https?:\/\//.test(trimmedValue)) {
    return trimmedValue;
  }

  return `http://${trimmedValue}`;
}

const apiProxyBaseUrl = normalizeApiProxyBaseUrl(process.env.API_PROXY_BASE_URL);

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    if (!apiProxyBaseUrl) {
      return [];
    }

    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyBaseUrl}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
