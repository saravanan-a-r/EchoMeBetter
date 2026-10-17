import { describe, expect, test } from '@jest/globals';
import { countReplace, NO_OFFERS_YET, parseOfferRecord, type OfferRecord } from '../../../shared/siteAccessOffer';

/** Replaces `times` rewrites in a row, as a user on a website who has not allowed it yet; the replaces that offered. */
function replaces(times: number, record: OfferRecord = NO_OFFERS_YET, eligible = true): { record: OfferRecord; offeredAt: number[] } {
  const offeredAt: number[] = [];
  for (let i = 0; i < times; i++) {
    const next = countReplace(record, eligible);
    record = next.record;
    if (next.offer) offeredAt.push(record.replaces);
  }
  return { record, offeredAt };
}

describe('when the hold and shortcuts offer comes', () => {
  test('after the second Replace, once more ten Replaces later, then never', () => {
    expect(replaces(40).offeredAt).toEqual([2, 12]);
  });

  test('not after the first Replace', () => {
    expect(replaces(1).offeredAt).toEqual([]);
  });

  test('a Replace where an offer makes no sense is counted but does not offer; the next one that can, does', () => {
    const welcome = replaces(2, NO_OFFERS_YET, false);
    expect(welcome.offeredAt).toEqual([]);
    expect(welcome.record).toEqual({ replaces: 2, offered: 0 });
    expect(replaces(1, welcome.record).offeredAt).toEqual([3]);
  });

  test('stored records are read defensively', () => {
    expect(parseOfferRecord(undefined)).toEqual(NO_OFFERS_YET);
    expect(parseOfferRecord({ replaces: -1, offered: 'x' })).toEqual(NO_OFFERS_YET);
    expect(parseOfferRecord({ replaces: 5, offered: 1 })).toEqual({ replaces: 5, offered: 1 });
  });
});
