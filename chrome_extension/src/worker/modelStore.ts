/**
 * The downloaded model, in the extension's Origin Private File System.
 *
 *   model/installed.json   which model is installed and where it came from;
 *                          written last, so it exists only once every file
 *                          has been downloaded and verified
 *   model/<sha256>         each model file, named by its content hash
 *
 * Naming files by their hash keeps every state simple: a half-downloaded
 * file is a file shorter than model.json says (the next download resumes
 * it), and replacing a model never touches the installed one's files until
 * the new one is complete.
 *
 * File contents are read and written through synchronous access handles,
 * the fastest way to move hundreds of megabytes in and out of OPFS; only
 * dedicated workers have them, so this runs in the inference worker.
 * Directory operations (open, list, remove) are asynchronous everywhere.
 */
import { EchoError } from '../shared/errors';
import { modelFiles, parseModelManifest, type ModelFile, type ModelManifest } from '../engine/manifest';

/** The parts of FileSystemSyncAccessHandle this module uses. */
export interface SyncAccessHandleLike {
  read(buffer: Uint8Array, options: { at: number }): number;
  write(buffer: Uint8Array, options: { at: number }): number;
  getSize(): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

export interface FileHandleLike {
  createSyncAccessHandle(): Promise<SyncAccessHandleLike>;
}

export interface DirectoryHandleLike {
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<DirectoryHandleLike>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileHandleLike>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
  keys(): AsyncIterable<string>;
}

export interface StoredModel {
  readonly sourceUrl: string;
  readonly manifest: ModelManifest;
}

const FOLDER = 'model';
const RECORD = 'installed.json';
/** Large reads are split so loading can report progress. */
const READ_CHUNK = 16 * 1024 * 1024;

function isNotFound(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'NotFoundError';
}

async function withHandle<T>(file: FileHandleLike, use: (handle: SyncAccessHandleLike) => T): Promise<T> {
  const handle = await file.createSyncAccessHandle();
  try {
    return use(handle);
  } finally {
    handle.close();
  }
}

function readAll(handle: SyncAccessHandleLike, onBytes?: (count: number) => void): Uint8Array {
  const size = handle.getSize();
  const out = new Uint8Array(size);
  for (let at = 0; at < size; ) {
    const count = handle.read(out.subarray(at, Math.min(size, at + READ_CHUNK)), { at });
    if (count <= 0) throw new Error(`short read at byte ${at} of ${size}`);
    at += count;
    onBytes?.(count);
  }
  return out;
}

export class ModelStore {
  private constructor(private readonly folder: DirectoryHandleLike) {}

  static async open(root: DirectoryHandleLike): Promise<ModelStore> {
    return new ModelStore(await root.getDirectoryHandle(FOLDER, { create: true }));
  }

  /** The names `manifest`'s model occupies, record included. */
  static namesOf(manifest: ModelManifest): Set<string> {
    return new Set([RECORD, ...modelFiles(manifest).map((file) => file.sha256)]);
  }

  /** The installed model: its record parses and every file it names is on disk at full size. */
  async installed(): Promise<StoredModel | null> {
    const file = await this.fileHandle(RECORD, false);
    if (!file) return null;
    let model: StoredModel;
    try {
      const record = JSON.parse(new TextDecoder().decode(await withHandle(file, (handle) => readAll(handle)))) as {
        sourceUrl?: unknown;
        manifest?: unknown;
      };
      if (typeof record.sourceUrl !== 'string') return null;
      model = { sourceUrl: record.sourceUrl, manifest: parseModelManifest(record.manifest) };
    } catch {
      // Unreadable or from an incompatible version: as good as absent; the next download replaces it.
      return null;
    }
    for (const modelFile of modelFiles(model.manifest)) {
      if ((await this.size(modelFile)) !== modelFile.bytes) return null;
    }
    return model;
  }

  /** Bytes of `file` on disk; 0 when it is absent. */
  async size(file: ModelFile): Promise<number> {
    const handle = await this.fileHandle(file.sha256, false);
    return handle ? withHandle(handle, (access) => access.getSize()) : 0;
  }

  /** Open `file` for writing at byte offsets, creating it if needed. The caller must close the handle. */
  async openForWrite(file: ModelFile): Promise<SyncAccessHandleLike> {
    const handle = await this.fileHandle(file.sha256, true);
    return handle!.createSyncAccessHandle();
  }

  /** The whole of `file`, which must be on disk at its full size. */
  async read(file: ModelFile, onBytes?: (count: number) => void): Promise<Uint8Array> {
    const handle = await this.fileHandle(file.sha256, false);
    if (!handle) throw new EchoError('MODEL_NOT_DOWNLOADED', { file: file.path });
    return withHandle(handle, (access) => {
      if (access.getSize() !== file.bytes) throw new EchoError('MODEL_NOT_DOWNLOADED', { file: file.path, reason: 'incomplete' });
      return readAll(access, onBytes);
    });
  }

  async remove(file: ModelFile): Promise<void> {
    await this.removeEntry(file.sha256);
  }

  /** Mark the model complete. Call only after every file has been verified. */
  async commit(sourceUrl: string, rawManifest: unknown): Promise<void> {
    const bytes = new TextEncoder().encode(JSON.stringify({ sourceUrl, manifest: rawManifest }));
    const handle = await this.fileHandle(RECORD, true);
    await withHandle(handle!, (access) => {
      access.truncate(0);
      access.write(bytes, { at: 0 });
      access.flush();
    });
  }

  /** Delete every entry not named in `keep`. */
  async prune(keep: ReadonlySet<string>): Promise<void> {
    const names: string[] = [];
    for await (const name of this.folder.keys()) names.push(name);
    for (const name of names) {
      if (!keep.has(name)) await this.removeEntry(name);
    }
  }

  async clear(): Promise<void> {
    await this.prune(new Set());
  }

  private async fileHandle(name: string, create: boolean): Promise<FileHandleLike | null> {
    try {
      return await this.folder.getFileHandle(name, { create });
    } catch (error) {
      if (!create && isNotFound(error)) return null;
      throw error;
    }
  }

  private async removeEntry(name: string): Promise<void> {
    try {
      await this.folder.removeEntry(name, { recursive: true });
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }
}
