/**
 * An in-memory Origin Private File System with the behaviour the model store
 * relies on: synchronous access handles that lock their file, NotFoundError
 * for missing entries, and positional reads and writes.
 */
import type { DirectoryHandleLike, FileHandleLike, SyncAccessHandleLike } from '../../worker/modelStore';

function notFound(name: string): DOMException {
  return new DOMException(`${name} not found`, 'NotFoundError');
}

export class MemoryFile implements FileHandleLike {
  data = new Uint8Array(0);
  locked = false;

  async createSyncAccessHandle(): Promise<SyncAccessHandleLike> {
    if (this.locked) throw new DOMException('file is locked by another access handle', 'NoModificationAllowedError');
    this.locked = true;
    const file = this;
    let open = true;
    const check = () => {
      if (!open) throw new DOMException('access handle is closed', 'InvalidStateError');
    };
    return {
      read(buffer, { at }) {
        check();
        const slice = file.data.subarray(at, at + buffer.byteLength);
        buffer.set(slice);
        return slice.byteLength;
      },
      write(buffer, { at }) {
        check();
        const end = at + buffer.byteLength;
        if (end > file.data.byteLength) {
          const grown = new Uint8Array(end);
          grown.set(file.data);
          file.data = grown;
        }
        file.data.set(buffer, at);
        return buffer.byteLength;
      },
      getSize() {
        check();
        return file.data.byteLength;
      },
      truncate(size) {
        check();
        const next = new Uint8Array(size);
        next.set(file.data.subarray(0, size));
        file.data = next;
      },
      flush() {
        check();
      },
      close() {
        open = false;
        file.locked = false;
      },
    };
  }
}

export class MemoryDirectory implements DirectoryHandleLike {
  readonly entries = new Map<string, MemoryFile | MemoryDirectory>();

  async getDirectoryHandle(name: string, options: { create?: boolean } = {}): Promise<MemoryDirectory> {
    const entry = this.entries.get(name);
    if (entry instanceof MemoryDirectory) return entry;
    if (entry || !options.create) throw notFound(name);
    const directory = new MemoryDirectory();
    this.entries.set(name, directory);
    return directory;
  }

  async getFileHandle(name: string, options: { create?: boolean } = {}): Promise<MemoryFile> {
    const entry = this.entries.get(name);
    if (entry instanceof MemoryFile) return entry;
    if (entry || !options.create) throw notFound(name);
    const file = new MemoryFile();
    this.entries.set(name, file);
    return file;
  }

  async removeEntry(name: string): Promise<void> {
    const entry = this.entries.get(name);
    if (!entry) throw notFound(name);
    if (entry instanceof MemoryFile && entry.locked) throw new DOMException(`${name} is open`, 'NoModificationAllowedError');
    this.entries.delete(name);
  }

  async *keys(): AsyncIterable<string> {
    for (const name of [...this.entries.keys()]) yield name;
  }

  /** Test convenience: the names in this directory, sorted. */
  names(): string[] {
    return [...this.entries.keys()].sort();
  }

  /** Test convenience: a file's bytes. */
  file(name: string): MemoryFile {
    const entry = this.entries.get(name);
    if (!(entry instanceof MemoryFile)) throw notFound(name);
    return entry;
  }
}
