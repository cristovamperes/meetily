import { invoke } from '@tauri-apps/api/core';
import { join } from '@tauri-apps/api/path';

// Use the backend's isolated data directory, not Tauri's default appDataDir.
export async function storePath(filename: string): Promise<string> {
  return join(await invoke<string>('get_database_directory'), filename);
}
