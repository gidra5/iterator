import {
  Falsy,
  Flat,
  Flattable,
  FromEntries,
  Product,
  RecordEntry,
  RecordKey,
  ToEntries,
  TupleN,
  Unzip,
  Zip,
  ZipLongest,
} from './types';
import { identity, isEqual, spread } from './utils';
export { spread } from './utils';
const sorter = (a: number, b: number): number => a - b;

const done = { done: true } as { done: true; value: undefined };

class RangeIteratorUp implements Iterator<number> {
  constructor(
    private value: number,
    private end: number,
    private step: number
  ) {}

  next() {
    const value = this.value;
    if (value >= this.end) return done;
    this.value += this.step;
    return { value };
  }
}

class RangeIteratorDown implements Iterator<number> {
  constructor(
    private value: number,
    private end: number,
    private step: number
  ) {}

  next() {
    const value = this.value;
    if (value <= this.end) return done;
    this.value += this.step;
    return { value };
  }
}

class RepeatIterator<T> implements Iterator<T> {
  constructor(private value: T) {}

  next() {
    return this as any;
  }
}

class MapIterator<T, U> implements Iterator<U> {
  constructor(private base: Iterator<T>, private mapper: (x: T) => U) {}

  next() {
    const value = this.base.next();
    if (value.done) return done;
    return { done: false as const, value: this.mapper(value.value) };
  }
}

class FilterIterator<T> implements Iterator<T> {
  constructor(
    private base: Iterator<T>,
    private predicate: (x: T) => boolean
  ) {}

  next() {
    for (let value; !(value = this.base.next()).done; ) {
      if (!this.predicate(value.value)) continue;
      return value;
    }
    return done;
  }
}

class FlatMapIterator<T, U> implements Iterator<T> {
  private nested: Iterator<T> | null = null;

  constructor(
    private base: Iterator<U>,
    private mapper: (x: U) => Iterable<T>
  ) {}

  next(): IteratorResult<T> {
    if (!this.nested) {
      const next = this.base.next();
      if (next.done) return next;
      this.nested = this.mapper(next.value)[Symbol.iterator]();
    }

    const next = this.nested!.next();
    if (!next.done) return next;

    this.nested = null;
    return this.next();
  }
}

class AccumulateIterator<T> implements Iterator<T> {
  constructor(
    private base: Iterator<T>,
    private reducer: (acc: T | undefined, input: T) => T,
    private acc: T | undefined
  ) {}

  next() {
    if (this.acc === undefined) {
      const next = this.base.next();
      if (next.done) return done;
      this.acc = next.value;
      return next;
    }

    const next = this.base.next();
    if (next.done) return done;

    this.acc = this.reducer(this.acc, next.value);
    return { done: false, value: this.acc };
  }
}

class CachedIterator<T> implements Iterator<T> {
  private index = 0;

  constructor(
    private base: Iterator<T>,
    private buffer: T[],
    private consumed: boolean,
    private setConsumed: () => void
  ) {}

  next() {
    if (this.index < this.buffer.length) {
      return { done: false, value: this.buffer[this.index++] };
    }

    if (this.consumed) return done;

    const item = this.base.next();

    if (item.done) {
      this.setConsumed();
      this.consumed = true;
      return done;
    }

    this.buffer.push(item.value);
    this.index++;
    return item;
  }
}

class SamplesIterator<T> implements Iterator<T> {
  private buffer: IteratorResult<T>[];
  private consumed = false;

  constructor(private base: Iterator<T>, private bufferSize: number) {
    this.buffer = new Array();
  }

  next() {
    // TODO: maybe we can avoid pulling bufferSize items from base iterator
    // with some probability magic
    // https://www.johndcook.com/blog/2010/04/06/subfactorial/
    while (!this.consumed && this.buffer.length < this.bufferSize) {
      const next = this.base.next();
      this.consumed = !!next.done;
      if (next.done) break;
      this.buffer.push(next);
    }

    if (this.buffer.length === 0) return done;
    if (this.buffer.length === 1) return this.buffer.pop()!;

    const index = Math.floor(Math.random() * this.buffer.length);
    const next = this.base.next();
    const value = !next.done
      ? this.buffer.splice(index, 1, next)[0]
      : this.buffer.splice(index, 1)[0];
    return value;
  }
}

// TODO: add docs
class _Iterator<T> implements Iterable<T> {
  private constructor(private getIterator: () => Iterator<T>) {}

  [Symbol.iterator]() {
    return this.getIterator();
  }

  // base methods

  cached() {
    let it = this.getIterator();
    let buffer: T[] = [];
    let consumed = false;
    const setConsumed = () => (consumed = true);
    return new _Iterator<T>(() =>
      consumed
        ? buffer[Symbol.iterator]()
        : new CachedIterator(it, buffer, consumed, setConsumed)
    );
  }

  consumable() {
    const it = this.getIterator();
    return new _Iterator(() => it);
  }

  accumulate(reducer: (acc: T, input: T) => T): _Iterator<T>;
  accumulate<U>(reducer: (acc: U, input: T) => U, initial?: U): _Iterator<U>;
  accumulate(reducer: (acc: any, input: T) => any, initial?: any) {
    return new _Iterator(
      () => new AccumulateIterator(this.getIterator(), reducer, initial)
    );
  }

  reduce(reducer: (acc: T | undefined, input: T) => T): T | undefined;
  reduce<U>(reducer: (acc: U, input: T) => U, initial?: U): U;
  reduce(reducer: (acc: any, input: T) => any, initial?: any) {
    let it = this.getIterator();
    let acc = initial;
    if (acc === undefined) {
      const next = it.next();
      if (next.done) return;
      acc = next.value;
    }

    for (let item; !(item = it.next()).done; ) {
      acc = reducer(acc, item.value);
    }
    return acc;
  }

  filter(pred: (x: T) => boolean): _Iterator<T>;
  filter<U extends T>(pred: (x: T) => x is U): _Iterator<U>;
  filter(pred: (x: T) => boolean) {
    return new _Iterator(() => new FilterIterator(this.getIterator(), pred));
  }

  map<U>(map: (x: T) => U) {
    return new _Iterator<U>(() => new MapIterator(this.getIterator(), map));
  }

  flatMap<U>(map: (x: T) => Iterable<U>) {
    return new _Iterator<U>(() => new FlatMapIterator(this.getIterator(), map));
  }

  flat<U, D extends number>(
    this: _Iterator<Flattable<U>>,
    depth: D
  ): _Iterator<Flat<T, D>>;
  flat<U>(this: _Iterator<Flattable<U>>): _Iterator<Flat<T, 1>>;
  flat<U, V extends Iterable<U>>(this: _Iterator<V>, depth = 1) {
    if (depth <= 0) return this;
    if (depth === 1) return this.flatMap(identity);
    const it = this;
    const gen = function* () {
      for (const item of it) {
        if (!item) yield item;
        else if (typeof item !== 'object') yield item;
        else if (!item[Symbol.iterator]) yield item;
        else yield* _Iterator.iter(item).flat(depth - 1);
      }
    };
    return new _Iterator(gen);
  }

  takeWhile(pred: (x: T) => boolean) {
    const it = this;
    const gen = function* () {
      for (const item of it) {
        if (!pred(item)) return;
        yield item;
      }
    };
    return new _Iterator(gen);
  }

  skipWhile(pred: (x: T) => boolean) {
    const it = this;
    const gen = function* () {
      let started = false;
      for (const item of it) {
        if (!started && !pred(item)) started = true;
        if (started) yield item;
      }
    };
    return new _Iterator(gen);
  }

  take(count: number) {
    const it = this;
    const gen = function* () {
      if (count <= 0) return;
      for (const item of it) {
        yield item;
        count--;
        if (count <= 0) return;
      }
    };
    return new _Iterator(gen);
  }

  skip(count: number) {
    if (count <= 0) return this;
    const it = this;
    const gen = () => {
      const iterator = it.getIterator();
      while (count > 0 && !iterator.next().done) count--;
      return iterator;
    };
    return new _Iterator(gen);
  }

  cycle() {
    const it = this.cached();
    const gen = function* () {
      if (it.isEmpty()) return;
      while (true) {
        yield* it;
      }
    };
    return new _Iterator(gen);
  }

  chunks(size: number) {
    const it = this.consumable();
    const gen = function* () {
      while (true) {
        const chunk = it.take(size).toArray();
        if (chunk.length === 0) return;
        yield chunk;
      }
    };
    return new _Iterator(gen);
  }

  // for streamSize <= bufferSize will produce "true" random permutation of its items
  samples(bufferSize = 20) {
    return new _Iterator(
      () => new SamplesIterator(this.getIterator(), bufferSize)
    );
  }

  sorted(_compare?: (a: T, b: T) => number): _Iterator<T> {
    // sort number values, then string values, then by type names
    const defaultCompare = (a: T, b: T) =>
      typeof a === 'number' && typeof b === 'number'
        ? a - b
        : typeof a === 'string' && typeof b === 'string'
        ? a.localeCompare(b)
        : (typeof a).localeCompare(typeof b);
    const compare = _compare ?? defaultCompare;
    const buffer: T[] = [];
    let done = false;
    const findInsertIndex = (x: T) => {
      let low = 0;
      let high = buffer.length;
      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (compare!(buffer[mid], x) < 0) low = mid + 1;
        else high = mid;
      }
      return low;
    };
    const it = this;
    const gen = function* () {
      // that's O(n+log(n)) btw
      // collecting all items to a sorted array
      // and then yielding it
      for (const item of it) {
        buffer.splice(findInsertIndex(item), 0, item);
      }
      done = true;
      yield* buffer;
    };
    return new _Iterator(() => (done ? buffer[Symbol.iterator]() : gen()));
  }

  // static methods

  static rotations<T>(items: T[]) {
    return _Iterator.chain(items, items).window(items.length);
  }

  static empty<T>() {
    return _Iterator.iter<T>([]);
  }

  static product<T extends Iterable<unknown>[]>(
    ...args: T
  ): _Iterator<Product<T>>;
  static product<T extends Iterable<unknown>[]>(...args: T) {
    if (args.length === 0) return _Iterator.empty();
    if (args.length === 1) return _Iterator.iter(args[0]).map((x) => [x]);
    const [head, ...rest] = args.map((it) => _Iterator.iter(it).cached());
    const gen = function* () {
      for (const item of head) {
        yield* _Iterator.product(...rest).map((tuple) => [item, ...tuple]);
      }
    };
    return new _Iterator(gen);
  }

  static zip<T extends Iterable<unknown>[]>(...iterables: T): _Iterator<Zip<T>>;
  static zip<T extends Iterable<unknown>[]>(...iterables: T) {
    const gen = function* () {
      const iterators = iterables.map((it) => it[Symbol.iterator]());
      while (true) {
        const items = iterators.map((it) => it.next());
        if (items.some((x) => x.done)) return;
        yield items.map((x) => x.value) as Zip<T>;
      }
    };
    return new _Iterator(gen);
  }

  static zipLongest<T extends Iterable<unknown>[]>(
    ...iterables: T
  ): _Iterator<ZipLongest<T>>;
  static zipLongest<T extends Iterable<unknown>[]>(...iterables: T) {
    const gen = function* () {
      const iterators = iterables.map((it) => it[Symbol.iterator]());
      while (true) {
        const items = iterators.map((it) => it.next());
        if (items.every((x) => x.done)) return;
        yield items.map((x) => x.value) as ZipLongest<T>;
      }
    };
    return new _Iterator(gen);
  }

  static chain<T>(...iterables: Iterable<T>[]) {
    return _Iterator.iter(iterables).flat();
  }

  static random() {
    const gen = function* () {
      while (true) yield Math.random();
    };
    return new _Iterator(gen);
  }

  static randomItems<T>(items: T[]) {
    return _Iterator.random().map((x) => items[Math.floor(x * items.length)]);
  }

  static shuffle<T>(items: T[]) {
    return _Iterator.iter(items).samples(items.length).toArray();
  }

  static randomInRange(min: number, max: number) {
    const delta = max - min;
    return _Iterator.random().map((x) => x * delta + min);
  }

  static randomDistribution(cpdf: (x: number) => number) {
    const dx = 1e-4;
    return _Iterator.random().map((y) => {
      // newton's method
      let x = 0;
      let dy = Infinity;
      while (Math.abs(dy) > dx) {
        const yEstimate = cpdf(x);
        dy = yEstimate - y;
        x -= (dy * dx) / (cpdf(x + dx) - yEstimate);
      }
      return x;
    });
  }

  // just copypaste of https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#partitions
  static partitions<T>(items: T[]) {
    return _Iterator.subsets(_Iterator.range(1, items.length)).map((x) => {
      return _Iterator
        .iter([0, ...x])
        .zip([...x, items.length])
        .map(([a, b]) => items.slice(a, b))
        .toArray();
    });
  }

  static combinations<T>(items: T[], size = items.length) {
    const range = _Iterator.natural(items.length).toArray();

    return _Iterator.permutation(range, size).filterMap((indices) => {
      if (isEqual(indices, indices.slice().sort(sorter)))
        return indices.map((i) => items[i]);
    });
  }

  static combinationsWithReplacement<T>(items: T[], size = items.length) {
    return _Iterator
      .natural(items.length)
      .power(size)
      .filterMap((indices) => {
        if (isEqual(indices, indices.slice().sort(sorter)))
          return indices.map((i) => items[i]);
      });
  }

  static permutation<T>(items: T[], size = items.length): _Iterator<T[]> {
    const gen = function* () {
      if (size > items.length) return;
      if (size === 0) yield [];
      for (let i = 0; i < items.length; i++) {
        const x = items[i];
        const rest = items.filter((_, _i) => _i !== i);
        const restPermutations = _Iterator.permutation(rest, size - 1);
        yield* restPermutations.map((xs) => [x, ...xs]);
      }
    };
    return new _Iterator<T[]>(gen);
  }

  static subsets<T>(items: Iterable<T>): _Iterator<T[]> {
    const it = _Iterator.iter(items).cached();
    const gen = function* () {
      yield it.toArray();
      for (const x of it) {
        const remainingItems = it.filter((_x) => _x !== x);
        yield* _Iterator.subsets(remainingItems);
      }
    };
    return new _Iterator(gen);
  }

  static subslices<T>(items: T[]) {
    const it = _Iterator.iter(items);
    return _Iterator
      .natural(items.length)
      .map((size) => [it.take(size), size] as [_Iterator<T>, number])
      .flatMap(([it, size]) =>
        _Iterator.natural(size).map((offset) => it.skip(offset).toArray())
      );
  }

  static prefixes<T>(items: T[]) {
    const it = _Iterator.iter(items);
    return _Iterator
      .natural(items.length)
      .map((size) => it.take(size).toArray());
  }

  static roundRobin<T>(...iterables: Iterable<T>[]) {
    const gen = function* () {
      const iterators = iterables.map((it) => it[Symbol.iterator]());
      while (true) {
        for (const it of [...iterators]) {
          const item = it.next();
          if (item.done) {
            iterators.splice(iterators.indexOf(it), 1);
            if (iterators.length === 0) return;
            continue;
          }
          yield item.value;
        }
      }
    };
    return new _Iterator(gen);
  }

  static repeat<T>(x: T) {
    return new _Iterator<T>(() => new RepeatIterator(x));
  }

  static range(start: number, end: number, step = 1) {
    const genUp = () => new RangeIteratorUp(start, end, step);
    const genDown = () => new RangeIteratorDown(start, end, step);
    return new _Iterator<number>(step > 0 ? genUp : genDown);
  }

  static natural(end: number = Infinity) {
    return _Iterator.range(0, Math.max(0, end));
  }

  static iter<T>(it: Iterable<T> | Iterator<T>): _Iterator<T> {
    if (it instanceof this) return it;
    if (typeof it === 'object' && 'next' in it) return new _Iterator(() => it);
    return new _Iterator(() => it[Symbol.iterator]());
  }

  static iterEntries<T extends Record<RecordKey, any>>(
    x: T
  ): _Iterator<ToEntries<T>>;
  static iterEntries<K extends RecordKey, T>(x: Record<K, T>) {
    const gen = function* () {
      for (const key in x) yield [key, x[key]] as [key: K, value: T];
    };
    return new _Iterator(gen);
  }

  static iterKeys<K extends RecordKey>(x: Record<K, unknown>) {
    const gen = function* () {
      for (const key in x) yield key;
    };
    return new _Iterator(gen);
  }

  static iterValues<T>(x: Record<RecordKey, T>) {
    const gen = function* () {
      for (const key in x) yield x[key];
    };
    return new _Iterator(gen);
  }

  // collection methods

  toArray() {
    return Array.from(this);
  }

  toObject<U extends RecordEntry>(this: _Iterator<U>): FromEntries<U> {
    return Object.fromEntries(this) as FromEntries<U>;
  }

  toMap<K, V>(this: _Iterator<[K, V]>) {
    return new Map(this);
  }

  toSet() {
    return new Set(this);
  }

  collect(f?: (x: T) => void) {
    for (const item of this) f?.(item);
  }

  // derived methods

  chain(...rest: Iterable<T>[]) {
    return _Iterator.chain(this, ...rest);
  }

  append(...next: T[]) {
    return _Iterator.chain(this, next);
  }

  prepend(...prev: T[]) {
    return _Iterator.chain(prev, this);
  }

  product<T extends Iterable<unknown>[]>(...args: Iterable<T>[]) {
    return _Iterator.product(this, ...args);
  }

  join(this: _Iterator<string>, separator = ',') {
    return this.reduce((acc, x) => acc + separator + x) ?? '';
  }

  max(this: _Iterator<number>): number;
  max(this: _Iterator<number>, accumulate: true): _Iterator<number>;
  max(accessor: (x: T) => number): number;
  max(accessor: (x: T) => number, accumulate: true): _Iterator<number>;
  max(...args: any[]) {
    const accessor = typeof args[0] === 'function' ? args[0] : undefined;
    const accumulate = typeof args[0] === 'boolean' ? args[0] : args[1];
    const reducer: any = accessor
      ? (acc: number, x: T) => Math.max(acc, accessor(x))
      : Math.max;
    const constructor = accumulate ? this.accumulate : this.reduce;
    const initial = !accumulate || accessor ? -Infinity : undefined;
    return constructor.call(this, reducer, initial);
  }

  min(this: _Iterator<number>): number;
  min(this: _Iterator<number>, accumulate: true): _Iterator<number>;
  min(accessor: (x: T) => number): number;
  min(accessor: (x: T) => number, accumulate: true): _Iterator<number>;
  min(...args: any[]) {
    const accessor = typeof args[0] === 'function' ? args[0] : undefined;
    const accumulate = typeof args[0] === 'boolean' ? args[0] : args[1];
    const reducer: any = accessor
      ? (acc: number, x: T) => Math.min(acc, accessor(x))
      : Math.min;
    const constructor = accumulate ? this.accumulate : this.reduce;
    const initial = !accumulate || accessor ? Infinity : undefined;
    return constructor.call(this, reducer, initial);
  }

  sum(this: _Iterator<number>): number;
  sum(this: _Iterator<number>, accumulate: true): _Iterator<number>;
  sum(accessor: (x: T) => number): number;
  sum(accessor: (x: T) => number, accumulate: true): _Iterator<number>;
  sum(...args: any[]) {
    const accessor = typeof args[0] === 'function' ? args[0] : undefined;
    const accumulate = typeof args[0] === 'boolean' ? args[0] : args[1];
    const reducer: any = accessor
      ? (acc: number, x: T) => acc + accessor(x)
      : (acc: number, x: number) => acc + x;
    const constructor = accumulate ? this.accumulate : this.reduce;
    const initial = !accumulate || accessor ? 0 : undefined;
    return constructor.call(this, reducer, initial);
  }

  mult(this: _Iterator<number>): number;
  mult(this: _Iterator<number>, accumulate: true): _Iterator<number>;
  mult(accessor: (x: T) => number): number;
  mult(accessor: (x: T) => number, accumulate: true): _Iterator<number>;
  mult(...args: any[]) {
    const accessor = typeof args[0] === 'function' ? args[0] : undefined;
    const accumulate = typeof args[0] === 'boolean' ? args[0] : args[1];
    const reducer: any = accessor
      ? (acc: number, x: T) => acc * accessor(x)
      : (acc: number, x: number) => acc * x;
    const constructor = accumulate ? this.accumulate : this.reduce;
    const initial = !accumulate || accessor ? 1 : undefined;
    return constructor.call(this, reducer, initial);
  }

  some(this: _Iterator<boolean>): boolean;
  some(pred: (x: T) => boolean): boolean;
  some(pred?: (x: T) => boolean) {
    if (!pred) return !(this as _Iterator<boolean>).filter(identity).isEmpty();
    return !this.filter(pred).isEmpty();
  }

  every(this: _Iterator<boolean>): boolean;
  every(pred: (x: T) => boolean): boolean;
  every(pred?: (x: T) => boolean) {
    if (!pred) return !this.some((x) => !x);
    return !this.some((x) => !pred(x));
  }

  avg(this: _Iterator<number>): _Iterator<number>;
  avg(accessor: (x: T) => number): _Iterator<number>;
  avg(accessor?: (x: T) => number) {
    return this.sum(accessor!, true)
      .enumerate()
      .map(spread((x, i) => x / (i + 1)));
  }

  mapValues<U, V, K extends RecordKey>(
    this: _Iterator<[K, U]>,
    map: (x: U) => V
  ) {
    return this.map(([key, value]) => [key, map(value)] as [K, V]);
  }

  filterValues<U, K extends RecordKey>(
    this: _Iterator<[K, U]>,
    pred: (x: U) => boolean
  ): _Iterator<[K, U]>;
  filterValues<U, V extends U, K extends RecordKey>(
    this: _Iterator<[K, U]>,
    pred: (x: U) => x is V
  ): _Iterator<[K, V]>;
  filterValues<U>(this: _Iterator<RecordEntry<U>>, pred: (x: U) => boolean) {
    return this.filter(([_, value]) => pred(value));
  }

  count() {
    return this.reduce((acc) => acc + 1, 0);
  }

  isEmpty() {
    return this.take(1).count() === 0;
  }

  power<N extends number>(n: N): _Iterator<TupleN<N, T>>;
  power(this: _Iterator<T>, n: number): _Iterator<T[]> {
    return _Iterator.product(..._Iterator.repeat(this).take(n));
  }

  zip<U extends Iterable<unknown>[]>(...iterables: U) {
    return _Iterator.zip<[_Iterator<T>, ...U]>(this, ...iterables);
  }

  zipLongest<U extends Iterable<unknown>[]>(...iterables: U) {
    return _Iterator.zipLongest<[_Iterator<T>, ...U]>(this, ...iterables);
  }

  spreadReduce<U extends unknown[], V>(
    this: _Iterator<U>,
    reducer: (acc: V, ...args: U) => V,
    initial: V
  ) {
    return this.reduce((acc, args) => reducer(acc, ...args), initial);
  }

  spreadAccumulate<U extends unknown[], V>(
    this: _Iterator<U>,
    reducer: (acc: V, ...args: U) => V,
    initial: V
  ) {
    return this.accumulate((acc, args) => reducer(acc, ...args), initial);
  }

  filterMap<U>(map: (x: T) => U | undefined | void) {
    return this.map(map).filter<U>((x): x is U => x !== undefined);
  }

  unzip<U extends unknown[]>(
    this: _Iterator<U>,
    size: U['length'] = Infinity
  ): Unzip<U> {
    const it = this.cached();
    size = Math.min(size, it.first()?.length ?? 0);
    return _Iterator
      .natural(size)
      .map((i) => it.map((x) => x[i]))
      .toArray() as Unzip<U>;
  }

  enumerate() {
    return this.zip(_Iterator.natural());
  }

  partition<U extends T>(pred: (x: T) => x is U): [_Iterator<U>, _Iterator<T>];
  partition(pred: (x: T) => boolean): [_Iterator<T>, _Iterator<T>];
  partition(pred: (x: T) => number): _Iterator<_Iterator<T>>;
  partition(pred: (x: T) => boolean | number): Iterable<_Iterator<T>> {
    const it = this.cached();
    const _pred = (x: T) => {
      const result = pred(x);
      if (typeof result === 'boolean') return result ? 1 : 0;
      return result;
    };
    return _Iterator.natural().map((i) => it.filter((x) => _pred(x) === i));
  }

  window<N extends number>(size: N) {
    const it = this.cached();
    const iterators = _Iterator.range(0, size).map((i) => it.skip(i));
    return _Iterator.zip(...iterators) as _Iterator<TupleN<N, T>>;
  }

  dot(this: _Iterator<number>, gen2: Iterable<number>) {
    return this.zip(gen2).sum((x) => x[0] * x[1]);
  }

  convolve(this: _Iterator<number>, kernel: number[]) {
    return this.window(kernel.length).map((x) => _Iterator.iter(x).dot(kernel));
  }

  padBefore(size: number, value: T) {
    return _Iterator.repeat(value).take(size).chain(this);
  }

  padAfter(size: number, value: T) {
    return this.chain(_Iterator.repeat(value).take(size));
  }

  pad(size: number, value: T) {
    return this.padBefore(size, value).padAfter(size, value);
  }

  nth(n: number) {
    return this.skip(n).take(1).toArray().pop();
  }

  first() {
    return this.nth(0);
  }

  last() {
    let item: T | undefined;
    this.collect((x) => (item = x));
    return item;
  }

  compact<T>(this: _Iterator<T | Falsy>) {
    return this.filter<T>(Boolean as unknown as (x: Falsy | T) => x is T);
  }

  dedupe(compare?: (a: T, b: T) => boolean) {
    let last: T | undefined;
    if (!compare) {
      return this.filterMap((x) => {
        if (last === x) return;
        last = x;
        return x;
      });
    }
    return this.filterMap((x) => {
      if (last !== undefined && compare(last, x)) return;
      last = x;
      return x;
    });
  }

  unique(compare?: (a: T, b: T) => boolean) {
    if (!compare) {
      const seen = new Set();
      return this.dedupe().filterMap((x) => {
        if (seen.has(x)) return;
        seen.add(x);
        return x;
      });
    }

    const seen: T[] = [];
    return this.dedupe(compare).filterMap((x) => {
      if (seen.some((y) => compare(x, y))) return;
      seen.push(x);
      return x;
    });
  }

  inOrder(compare: (a: T, b: T) => boolean) {
    return this.window(2).every(spread(compare));
  }

  inspect(callback: (x: T) => void) {
    return this.map((x) => (callback(x), x));
  }

  equals<U>(gen2: Iterable<U>, compare: (a: T, b: U) => boolean) {
    return this.zip(gen2).every(spread(compare));
  }

  group<U extends string>(key: (x: T) => U) {
    return this.reduce((map, x) => {
      const k = key(x);
      if (!(k in map)) map[k] = [];
      map[k].push(x);
      return map;
    }, {} as Record<U, T[]>);
  }

  groupToMap<U>(key: (x: T) => U) {
    return this.reduce((map, x) => {
      const k = key(x);
      if (!map.has(k)) map.set(k, []);
      map.get(k)!.push(x);
      return map;
    }, new Map<U, T[]>());
  }

  find(pred: (x: T) => boolean) {
    return this.filter(pred).first();
  }
}

export { _Iterator as Iterator };
