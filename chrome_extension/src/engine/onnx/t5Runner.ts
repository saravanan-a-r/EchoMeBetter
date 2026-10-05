/**
 * The "t5-cross-kv/1" I/O contract on top of onnxruntime-web sessions.
 *
 *   encoder.onnx  input_ids, attention_mask  →  cross_key.{i}, cross_value.{i}
 *   decoder.onnx  input_ids, encoder_attention_mask, past_key.{i}, past_value.{i},
 *                 cross_key.{i}, cross_value.{i}  →  logits, present_key.{i}, present_value.{i}
 *
 * The encoder produces every decoder layer's cross-attention keys/values
 * once; each decoder step then only processes the newest token, feeding the
 * previous step's `present_*` back in as `past_*`. Tensor names are checked
 * against the sessions on construction so a model exported for a different
 * contract fails immediately rather than mid-generation.
 */
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import type { Seq2SeqRunner, StepResult } from '../generation/greedyDecoder';

/** The constructor half of onnxruntime's API that the runner needs; injectable for tests. */
export interface TensorFactory {
  new (type: 'int64', data: BigInt64Array, dims: readonly number[]): Tensor;
  new (type: 'float32', data: Float32Array, dims: readonly number[]): Tensor;
}

export interface T5Geometry {
  readonly numDecoderLayers: number;
  readonly numHeads: number;
  readonly headDim: number;
}

export interface EncoderMemory {
  readonly mask: Tensor;
  readonly cross: Readonly<Record<string, Tensor>>;
}

export type DecoderCache = Readonly<Record<string, Tensor>>;

export class ContractError extends Error {
  constructor(message: string) {
    super(`model does not follow the t5-cross-kv/1 contract: ${message}`);
    this.name = 'ContractError';
  }
}

function requireNames(actual: readonly string[], expected: readonly string[], where: string): void {
  const missing = expected.filter((name) => !actual.includes(name));
  if (missing.length > 0) throw new ContractError(`${where} is missing ${missing.slice(0, 4).join(', ')}`);
}

export class T5Runner implements Seq2SeqRunner<EncoderMemory, DecoderCache> {
  private readonly crossNames: string[] = [];
  private readonly pastNames: [string, string][] = [];

  constructor(
    private readonly TensorCtor: TensorFactory,
    private readonly encoder: InferenceSession,
    private readonly decoder: InferenceSession,
    private readonly geometry: T5Geometry,
  ) {
    for (let i = 0; i < geometry.numDecoderLayers; i++) {
      this.crossNames.push(`cross_key.${i}`, `cross_value.${i}`);
      this.pastNames.push([`past_key.${i}`, `present_key.${i}`], [`past_value.${i}`, `present_value.${i}`]);
    }
    requireNames(encoder.inputNames, ['input_ids', 'attention_mask'], 'encoder inputs');
    requireNames(encoder.outputNames, this.crossNames, 'encoder outputs');
    requireNames(
      decoder.inputNames,
      ['input_ids', 'encoder_attention_mask', ...this.pastNames.map(([past]) => past), ...this.crossNames],
      'decoder inputs',
    );
    requireNames(decoder.outputNames, ['logits', ...this.pastNames.map(([, present]) => present)], 'decoder outputs');
  }

  private ids(values: readonly number[]): Tensor {
    return new this.TensorCtor('int64', BigInt64Array.from(values, (value) => BigInt(value)), [1, values.length]);
  }

  async encode(inputIds: readonly number[]): Promise<EncoderMemory> {
    const mask = new this.TensorCtor('int64', new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
    const outputs = await this.encoder.run({ input_ids: this.ids(inputIds), attention_mask: mask });
    const cross: Record<string, Tensor> = {};
    for (const name of this.crossNames) cross[name] = outputs[name]!;
    return { mask, cross };
  }

  initialCache(): DecoderCache {
    const { numHeads, headDim } = this.geometry;
    const cache: Record<string, Tensor> = {};
    for (const [past] of this.pastNames) cache[past] = new this.TensorCtor('float32', new Float32Array(0), [1, numHeads, 0, headDim]);
    return cache;
  }

  async step(tokenId: number, memory: EncoderMemory, cache: DecoderCache): Promise<StepResult<DecoderCache>> {
    const outputs = await this.decoder.run({
      input_ids: this.ids([tokenId]),
      encoder_attention_mask: memory.mask,
      ...cache,
      ...memory.cross,
    });
    const next: Record<string, Tensor> = {};
    for (const [past, present] of this.pastNames) next[past] = outputs[present]!;

    const logits = outputs.logits!;
    const vocab = logits.dims[logits.dims.length - 1]!;
    const data = logits.data as Float32Array;
    // Copy the last position's scores: processors mutate them in place.
    return { logits: data.slice(data.length - vocab), cache: next };
  }

  release(memory: EncoderMemory, cache: DecoderCache): void {
    for (const tensor of [...Object.values(memory.cross), ...Object.values(cache)]) tensor.dispose?.();
  }
}
