/**
 * The downloaded model, in the extension's Origin Private File System.
 *
 *   model/installed.json   where the model came from, its manifest and which
 *                          adapters are installed; rewritten only once every
 *                          file it names has been downloaded and verified
 *   model/<sha256>         each file (base or adapter), named by its content hash
 *
 * Naming files by their hash keeps every state simple: a half-downloaded
 * file is a file shorter than model.json says (the next download resumes
 * it), and adding or removing an adapter never touches the base's files.
 *
 * File contents are read and written through synchronous access handles,
 * the fastest way to move hundreds of megabytes in and out of OPFS; only
 * dedicated workers have them, so this runs in the inference worker.
 * Directory operations (open, list, remove) are asynchronous everywhere.
 */
import { EchoError } from '../shared/errors';
import { isStyleId, type StyleId } from '../shared/styles';
import { adapterFiles, baseFiles, parseModelManifest, type ModelFile, type ModelManifest } from '../engine/manifest';

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
  /** Installed adapters; each one's files are all on disk. */
  readonly adapters: readonly StyleId[];
}

/** Every file `model` occupies: the base and its installed adapters. */
export function filesOf(model: StoredModel): ModelFile[] {
  return [...baseFiles(model.manifest), ...model.adapters.flatMap((style) => adapterFiles(model.manifest.adapters[style]!))];
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

  /** The names `model` occupies, record included. */
  static namesOf(model: StoredModel): Set<string> {
    return new Set([RECORD, ...filesOf(model).map((file) => file.sha256)]);
  }

  async hasRecord(): Promise<boolean> {
    return (await this.fileHandle(RECORD, false)) !== null;
  }

  /**
   * The installed model: its record parses and every base file is on disk at
   * full size. Adapters whose files are not all there are left out.
   */
  async installed(): Promise<StoredModel | null> {
    const file = await this.fileHandle(RECORD, false);
    if (!file) return null;
    let sourceUrl: string;
    let manifest: ModelManifest;
    let recorded: unknown[];
    try {
      const record = JSON.parse(new TextDecoder().decode(await withHandle(file, (handle) => readAll(handle)))) as {
        sourceUrl?: unknown;
        manifest?: unknown;
        adapters?: unknown;
      };
      if (typeof record.sourceUrl !== 'string' || !Array.isArray(record.adapters)) return null;
      sourceUrl = record.sourceUrl;
      manifest = parseModelManifest(record.manifest);
      recorded = record.adapters;
    } catch {
      // Unreadable or from an incompatible version: as good as absent.
      return null;
    }
    if (!(await this.complete(baseFiles(manifest)))) return null;
    const adapters: StyleId[] = [];
    for (const style of recorded) {
      const adapter = isStyleId(style) ? manifest.adapters[style] : undefined;
      if (adapter && (await this.complete(adapterFiles(adapter)))) adapters.push(style as StyleId);
    }
    return { sourceUrl, manifest, adapters };
  }

  /** Whether every one of `files` is on disk at full size. */
  async complete(files: readonly ModelFile[]): Promise<boolean> {
    for (const modelFile of files) {
      if ((await this.size(modelFile)) !== modelFile.bytes) return false;
    }
    return true;
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

  /** Record `model` as installed. Call only after every file it names has been verified. */
  async commit(model: StoredModel): Promise<void> {
    const bytes = new TextEncoder().encode(JSON.stringify({ sourceUrl: model.sourceUrl, manifest: model.manifest, adapters: model.adapters }));
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
