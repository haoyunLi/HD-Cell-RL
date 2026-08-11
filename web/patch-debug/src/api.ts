import type { PatchManifest, PatchManifestEntry, PatchPayload } from "./types";

export function manifestUrlFromLocation(): string {
  const params = new URLSearchParams(window.location.search);
  return params.get("manifest") ?? new URL("patch_debug/manifest.json", document.baseURI).toString();
}

export async function loadManifest(url: string): Promise<PatchManifest> {
  return loadJson<PatchManifest>(url);
}

export async function loadPatch(manifestUrl: string, entry: PatchManifestEntry): Promise<PatchPayload> {
  const patchUrl = new URL(entry.file, new URL(manifestUrl, window.location.href));
  return loadJson<PatchPayload>(patchUrl.toString());
}

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}
