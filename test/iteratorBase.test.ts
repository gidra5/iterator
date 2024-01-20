import { describe, expect, vi } from 'vitest';
import { it, fc } from '@fast-check/vitest';
import Iterator from '../src';

describe.concurrent('iterator', () => {
  it.concurrent.prop([fc.array(fc.nat())])('Iterator is Iterable', (array) => {
    const iterator = Iterator.iter(array);
    let sum = 0;
    for (const value of iterator) {
      sum += value;
    }

    expect(sum).toEqual(array.reduce((a, b) => a + b, 0));
  });

  it.concurrent.prop([fc.array(fc.nat())])('Iterator.map', (array) => {
    const mapper = (value: number) => value * 2;
    expect(Iterator.iter(array).map(mapper).toArray()).toEqual(
      array.map(mapper)
    );
  });

  it.concurrent.prop([fc.array(fc.nat())])('Iterator.filter', (array) => {
    const predicate = (value: number) => value % 2 === 0;
    expect(Iterator.iter(array).filter(predicate).toArray()).toEqual(
      array.filter(predicate)
    );
  });

  it.concurrent.prop([
    fc.array(fc.array(fc.array(fc.array(fc.anything())))),
    fc.nat(),
  ])('Iterator.flat', (array, depth) => {
    const flatArray = array.flat(depth);
    expect(Iterator.iter(array).flat(depth).toArray()).toEqual(flatArray);
  });

  it.concurrent.prop([fc.array(fc.array(fc.array(fc.array(fc.anything()))))])(
    'Iterator.flat with no depth',
    (array) => {
      const flatArray = array.flat();
      expect(Iterator.iter(array).flat().toArray()).toEqual(flatArray);
    }
  );

  it.concurrent.prop([fc.array(fc.anything()), fc.nat()])(
    'Iterator.take',
    (array, size) => {
      const fn = vi.fn();
      const taken = Iterator.iter(array).inspect(fn).take(size).toArray();
      expect(taken).toEqual(array.slice(0, size));
      expect(fn).toHaveBeenCalledTimes(Math.min(array.length, size));
    }
  );

  it.concurrent.prop([fc.array(fc.nat())])('Iterator.takeWhile', (array) => {
    const pred = (value: number) => value < 3;
    const predicate = vi.fn(pred);
    const taken = Iterator.iter(array).takeWhile(predicate).toArray();
    const index = array.findIndex((value) => !pred(value));
    const expected = index !== -1 ? array.slice(0, index) : array;
    expect(taken).toEqual(expected);
    expect(predicate).toHaveBeenCalledTimes(
      index !== -1 ? index + 1 : array.length
    );
  });

  it.concurrent.prop([fc.array(fc.anything()), fc.nat()])(
    'Iterator.skip',
    (array, size) => {
      const skipped = Iterator.iter(array).skip(size).toArray();
      expect(skipped).toEqual(array.slice(size));
    }
  );

  it.concurrent.prop([fc.array(fc.nat())])('Iterator.skipWhile', (array) => {
    const pred = (value: number) => value < 3;
    const predicate = vi.fn(pred);
    const skipped = Iterator.iter(array).skipWhile(predicate).toArray();
    const index = array.findIndex((value) => !pred(value));
    const expected = index !== -1 ? array.slice(index) : [];
    expect(skipped).toEqual(expected);
    expect(predicate).toHaveBeenCalledTimes(
      index !== -1 ? index + 1 : array.length
    );
  });

  it.concurrent.prop([fc.array(fc.anything())])('Iterator.cached', (array) => {
    const fn = vi.fn();
    const fn2 = vi.fn();
    const iterator = Iterator.iter(array).inspect(fn).cached().inspect(fn2);
    expect(iterator.toArray()).toEqual(array);
    expect(iterator.toArray()).toEqual(array);
    expect(fn).toHaveBeenCalledTimes(array.length);
    expect(fn2).toHaveBeenCalledTimes(array.length * 2);
  });

  it.concurrent.prop([fc.array(fc.anything())])(
    'Iterator.consumable',
    (array) => {
      const fn = vi.fn();
      const fn2 = vi.fn();
      const iterator = Iterator.iter(array)
        .inspect(fn)
        .consumable()
        .inspect(fn2);
      expect(iterator.toArray()).toEqual(array);
      expect(iterator.toArray()).toEqual([]);
      expect(fn).toHaveBeenCalledTimes(array.length);
      expect(fn2).toHaveBeenCalledTimes(array.length);
    }
  );

  it.concurrent.prop([fc.array(fc.anything())])(
    'Iterator.cached undoes Iterator.consumable',
    (array) => {
      const fn = vi.fn();
      const fn2 = vi.fn();
      const fn3 = vi.fn();
      const iterator = Iterator.iter(array)
        .inspect(fn)
        .consumable()
        .inspect(fn2)
        .cached()
        .inspect(fn3);
      expect(iterator.toArray()).toEqual(array);
      expect(iterator.toArray()).toEqual(array);
      expect(fn).toHaveBeenCalledTimes(array.length);
      expect(fn2).toHaveBeenCalledTimes(array.length);
      expect(fn3).toHaveBeenCalledTimes(array.length * 2);
    }
  );

  it.concurrent.prop([fc.array(fc.anything())])(
    'Iterator.consumable undoes Iterator.cached',
    (array) => {
      const fn = vi.fn();
      const fn2 = vi.fn();
      const fn3 = vi.fn();
      const iterator = Iterator.iter(array)
        .inspect(fn)
        .cached()
        .inspect(fn2)
        .consumable()
        .inspect(fn3);
      expect(iterator.toArray()).toEqual(array);
      expect(iterator.toArray()).toEqual([]);
      expect(fn).toHaveBeenCalledTimes(array.length);
      expect(fn2).toHaveBeenCalledTimes(array.length);
      expect(fn3).toHaveBeenCalledTimes(array.length);
    }
  );

  it.concurrent.prop([fc.array(fc.anything())])('Iterator.cycle', (array) => {
    const iterator = Iterator.iter(array).cycle();
    expect(iterator.take(array.length * 2).toArray()).toEqual([
      ...array,
      ...array,
    ]);
  });

  it.concurrent.prop([fc.array(fc.anything())])(
    'Iterator.cycle caches',
    (array) => {
      const fn = vi.fn();
      const iterator = Iterator.iter(array).inspect(fn).cycle();
      iterator.take(array.length * 2).collect();
      expect(fn).toHaveBeenCalledTimes(array.length);
    }
  );

  // it.concurrent.prop([fc.array(fc.anything()), fc.nat()])(
  //   "Iterator.chunks",
  //   (array, size) => {
  //     const fn = vi.fn();
  //     const chunks = Iterator.iter(array).inspect(fn).chunks(size).toArray();
  //     const expected = Iterator.natural(Math.ceil(array.length / size))
  //       .map((i) => array.slice(i * size, (i + 1) * size))
  //       .toArray();
  //     expect(chunks).toEqual(expected);
  //     expect(fn).toHaveBeenCalledTimes(array.length);
  //   }
  // );

  it.concurrent.prop([
    fc.array(fc.anything(), { size: 'xlarge' }),
    fc.nat({ max: 100 }).filter((size) => size > 0),
  ])('Iterator.sample', (array, size) => {
    const fn = vi.fn();
    const samples = Iterator.iter(array).inspect(fn).samples(size);
    samples.take(1).toArray();
    expect(fn).toHaveBeenCalledTimes(Math.min(array.length, size + 1));
    expect(samples.count()).toEqual(array.length);
    // const samplesArray = samples.toArray();
    // expect(
    //   Iterator.permutation(array).some((value) => isEqual(value, samplesArray))
    // ).toBe(true);
  });
});
