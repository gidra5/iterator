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
export type IteratorInterface<T, TReturn = any, TNext = unknown> = Iterator<
  T,
  TReturn,
  TNext
>;
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
