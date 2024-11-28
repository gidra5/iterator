export type TupleN<
  N extends number,
  T = unknown,
  A extends unknown[] = []
> = number extends N
  ? T[]
  : A['length'] extends N
  ? A
  : TupleN<N, T, [...A, T]>;
export type Pred<A extends number> = TupleN<A> extends [...infer B, infer _]
  ? B['length']
  : A;

export type RecordKey = keyof any;
export type RecordEntry<T = unknown> = [RecordKey, T];
export type Falsy = false | 0 | '' | null | undefined;

export type Zip<T extends Iterable<unknown>[]> = T extends [
  Iterable<infer A>,
  ...infer B extends Iterable<unknown>[]
]
  ? [A, ...Zip<B>]
  : T extends []
  ? []
  : T extends Iterable<infer A>[]
  ? A[]
  : never;
export type ZipLongest<T extends Iterable<unknown>[]> = T extends [
  Iterable<infer A>,
  ...infer B extends Iterable<unknown>[]
]
  ? [A | undefined, ...ZipLongest<B>]
  : T extends []
  ? []
  : T extends Iterable<infer A>[]
  ? (A | undefined)[]
  : never;
export type Unzip<T extends unknown[]> = T extends [infer A, ...infer B]
  ? [Iterator<A>, ...Unzip<B>]
  : T extends []
  ? []
  : T extends (infer A)[]
  ? Iterator<A>[]
  : never;
export type Flat<T, D extends number> = D extends 0
  ? T
  : T extends Iterable<infer A>
  ? Flat<A, Pred<D>>
  : T;
export type Flattable<T> = T | Iterable<T>;
export type Product<T extends Iterable<unknown>[]> = T extends [
  Iterable<infer A>,
  ...infer B extends Iterable<unknown>[]
]
  ? [A, ...Product<B>]
  : T extends []
  ? []
  : T extends Iterable<infer A>[]
  ? A[]
  : never;
export type FromEntries<T extends RecordEntry> = {
  [K in T[0]]: T extends [K, infer V] ? V : never;
};
export type ToEntries<T> = { [K in keyof T]: [K, T[K]] }[keyof T];

declare global {
  interface Iterator<T> {
    map<R = T>(mapper: (value: T, index: number) => R): Iterator<R>;
    filter<R extends T = T>(
      filterer: (value: T, index: number) => value is R
    ): Iterator<R>;
    filter(filterer: (value: T, index: number) => unknown): Iterator<T>;
    take(limit: number): Iterator<T>;
    drop(limit: number): Iterator<T>;
    flatMap<R = T>(
      mapper: (value: T, index: number) => Iterator<R> | Iterable<R>
    ): Iterator<R>;
    forEach(fn: (value: T, index: number) => void): void;
    toArray(): T[];
    reduce(reducer: (previousValue: T, currentValue: T, index: number) => T): T;
    reduce(
      reducer: (previousValue: T, currentValue: T, index: number) => T,
      initialValue: T
    ): T;
    reduce<U>(
      reducer: (previousValue: U, currentValue: T, index: number) => U,
      initialValue: U
    ): U;
    some(fn: (value: T, index: number) => unknown): boolean;
    every(fn: (value: T, index: number) => unknown): boolean;
    find(fn: (value: T, index: number) => unknown): T | undefined;
    [Symbol.iterator](): Iterator<T>;
  }
  interface IteratorConstructor {
    readonly prototype: Iterator<any>;
    new <T = unknown>(): Iterator<T>;
    from<T>(iterable: Iterator<T> | Iterable<T>): Iterator<T>;
  }

  var Iterator: IteratorConstructor;
}
