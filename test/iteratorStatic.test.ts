import { describe, expect } from 'vitest';
import { it, fc, test } from '@fast-check/vitest';
import Iterator from '../src';

describe.concurrent('iterator', () => {
  const smallNat = fc.nat({ max: 20_000 });

  it.concurrent.prop([fc.array(fc.anything())])('Iterator.iter', (array) => {
    const iterator = Iterator.iter(array);
    expect(iterator.toArray()).toEqual(array);
  });

  it.concurrent.prop([smallNat])('Iterator.natural', (count) => {
    const iterator = Iterator.natural(count);
    expect(iterator.toArray()).toEqual([...Array(count).keys()]);
  });

  it.concurrent.prop([fc.array(fc.anything()), fc.array(fc.anything())])(
    'Iterator.zip',
    (array1, array2) => {
      const zipped = Iterator.zip(array1, array2).toArray();
      const length = Math.min(array1.length, array2.length);
      expect(zipped.length).toEqual(length);
      for (const i of Iterator.natural(length)) {
        expect(zipped[i]).toEqual([array1[i], array2[i]]);
      }
    }
  );

  it.concurrent.prop([fc.array(fc.array(fc.anything()), { minLength: 3 })])(
    'Iterator.zip with multiple iterators',
    (arrays) => {
      const zipped = Iterator.zip(...arrays).toArray();
      const length = Math.min(...arrays.map((arr) => arr.length));
      expect(zipped.length).toEqual(length);
      for (const i of Iterator.natural(length)) {
        expect(zipped[i]).toEqual(arrays.map((arr) => arr[i]));
      }
    }
  );

  it.concurrent.prop([fc.array(fc.array(fc.anything()), { minLength: 2 })])(
    'Iterator.chain',
    (arrays) => {
      const chained = Iterator.chain(...arrays).toArray();
      expect(chained).toEqual(arrays.flat());
    }
  );

  it.concurrent.prop([fc.object()])('Iterator.iterEntries', (object) => {
    const entries = Iterator.iterEntries(object).toArray();
    expect(entries).toEqual(Object.entries(object));
  });

  it.concurrent.prop([fc.object()])('Iterator.iterKeys', (object) => {
    const keys = Iterator.iterKeys(object).toArray();
    expect(keys).toEqual(Object.keys(object));
  });

  it.concurrent.prop([fc.object()])('Iterator.iterValues', (object) => {
    const values = Iterator.iterValues(object).toArray();
    expect(values).toEqual(Object.values(object));
  });

  // it.concurrent.prop([fc.array(fc.anything())])(
  //   "Iterator.permutation",
  //   (array) => {
  //     const permutation = Iterator.permutation(array);
  //     expect(permutation.count()).toEqual(factorial(array.length));
  //     expect(
  //       permutation.every(
  //         (value) =>
  //           value.length === array.length &&
  //           value.every((x) => array.includes(x)) &&
  //           array.every((x) => value.includes(x))
  //       )
  //     ).toBe(true);
  //   }
  // );

  test.concurrent('Iterator.range with start and stop', () => {
    const actual = Iterator.range(1, 5).toArray();
    expect(actual).toEqual([1, 2, 3, 4]);
  });

  test.concurrent('Iterator.range with start, stop and step', () => {
    const actual = Iterator.range(1, 6, 2).toArray();
    expect(actual).toEqual([1, 3, 5]);
  });

  test.concurrent('Iterator.range reverse', () => {
    const actual = Iterator.range(4, -1, -1).toArray();
    expect(actual).toEqual([4, 3, 2, 1, 0]);
  });

  test.concurrent('Iterator.range empty range', () => {
    const actual = Iterator.range(0, 0).toArray();
    expect(actual).toEqual([]);
  });

  test.concurrent('Iterator.range empty range with start > stop', () => {
    const actual = Iterator.range(1, 0).toArray();
    expect(actual).toEqual([]);
  });
});
