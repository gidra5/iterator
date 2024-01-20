import { describe, bench as _bench, afterEach } from 'vitest';
import { Iterator } from '../src';
import MetarhiaIterator from '@metarhia/iterator';

afterEach(async () => {
  // flush promises https://github.com/vitest-dev/vitest/issues/4497#
  await new Promise((res) => setImmediate(res));
});

const testIteration = (iterator: Iterable<number>) => {
  let sum = 0;
  for (const value of iterator) {
    sum += value;
  }
};
const sizes = Iterator.range(5, 6)
  .map((i) => 0.2 * i * 10 ** i)
  .map(Math.round);
const options = {
  /**
   * warmup time (milliseconds)
   * @default 100ms
   */
  warmupTime: 500,

  /**
   * warmup iterations
   * @default 5
   */
  warmupIterations: 25,
};

const bench = (name, fn, _options = options) => _bench(name, fn, _options);
Object.setPrototypeOf(bench, _bench);

for (const size of sizes) {
  const iteratorNat = Iterator.natural(size);
  const array = iteratorNat.toArray();
  const iterator = Iterator.iter(array);
  const metarhiaIterator = () => MetarhiaIterator.iter(array);
  const iterators = Iterator.iterEntries({ iterator, iteratorNat });
  const reducer = (acc: number, x: number) => acc + x;
  const map1 = (x: number) => x + 1;
  const map2 = (x: number) => x + 2;
  const filter1 = (x: number) => x % 2 === 0;
  const filter2 = (x: number) => x % 3 === 0;
  const flatMap1 = (x: number) => [x, x];
  const filterMap1 = (x: number) => (x % 2 === 0 ? x + 1 : undefined);
  const transforms = Iterator.iterEntries({
    map: [map1, map2],
    filter: [filter1, filter2],
    flatMap: [flatMap1],
    filterMap: [filterMap1],
  });

  for (const [name, _iterator] of iterators) {
    describe.concurrent(`${name} iteration cache - ${size}`, () => {
      bench(name, () => {
        testIteration(_iterator);
      });

      bench(`${name}.cached`, () => {
        testIteration(_iterator.cached());
      });

      bench(`${name}.consumable`, () => {
        testIteration(_iterator.consumable());
      });
    });
  }

  describe.concurrent(`iteration - ${size}`, () => {
    for (const [name, _iterator] of iterators) {
      bench(name, () => {
        testIteration(_iterator);
      });

      bench(`${name}.reduce`, () => {
        _iterator.reduce(reducer, 0);
      });

      bench(`${name}.sum`, () => {
        _iterator.sum();
      });
    }

    bench(`Array`, () => {
      testIteration(array);
    });

    bench(`Array.reduce`, () => {
      array.reduce(reducer, 0);
    });

    bench(`MetarhiaIterator`, () => {
      testIteration(metarhiaIterator());
    });

    bench(`MetarhiaIterator.reduce`, () => {
      metarhiaIterator().reduce(reducer, 0);
    });
  });

  for (const [transformName, transformFns] of transforms) {
    if (transformName === 'filterMap') {
      describe.concurrent(`iteration filterMap - ${size}`, () => {
        for (let [name, _iterator] of iterators) {
          bench(`${name}.filterMap`, () => {
            testIteration(_iterator.filterMap(transformFns[0]));
          });
          bench(`${name}.filterMap.reduce`, () => {
            _iterator.filterMap(transformFns[0]).reduce(reducer, 0);
          });
          bench(`${name}.filterMap.sum`, () => {
            _iterator.filterMap(transformFns[0]).sum();
          });
        }
        bench(`Array.filterMap`, () => {
          testIteration(
            array
              .map(transformFns[0])
              .filter((x): x is number => x !== undefined)
          );
        });
        bench(`Array.filterMap.reduce`, () => {
          array
            .map(transformFns[0])
            .filter((x): x is number => x !== undefined)
            .reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.filterMap`, () => {
          testIteration(metarhiaIterator().filterMap(transformFns[0]));
        });
        bench(`MetarhiaIterator.filterMap.reduce`, () => {
          metarhiaIterator().filterMap(transformFns[0]).reduce(reducer, 0);
        });
      });
    } else if (transformName === 'flatMap') {
      describe.concurrent(`iteration flatMap - ${size}`, () => {
        for (let [name, _iterator] of iterators) {
          bench(`${name}.flatMap`, () => {
            testIteration(_iterator.flatMap(transformFns[0]));
          });

          bench(`${name}.flatMap.reduce`, () => {
            _iterator.flatMap(transformFns[0]).reduce(reducer, 0);
          });

          bench(`${name}.flatMap.sum`, () => {
            _iterator.flatMap(transformFns[0]).sum();
          });
        }

        bench(`Array.flatMap`, () => {
          testIteration(array.flatMap(transformFns[0]));
        });

        bench(`Array.flatMap.reduce`, () => {
          array.flatMap(transformFns[0]).reduce(reducer, 0);
        });

        bench(`MetarhiaIterator.flatMap`, () => {
          testIteration(metarhiaIterator().flatMap(transformFns[0]));
        });

        bench(`MetarhiaIterator.flatMap.reduce`, () => {
          metarhiaIterator().flatMap(transformFns[0]).reduce(reducer, 0);
        });
      });
    } else if (transformName === 'filter') {
      describe.concurrent(`iteration filter - ${size}`, () => {
        for (let [name, _iterator] of iterators) {
          bench(`${name}.filter`, () => {
            testIteration(_iterator.filter(transformFns[0]));
          });
          bench(`${name}.filter.reduce`, () => {
            _iterator.filter(transformFns[0]).reduce(reducer, 0);
          });
          bench(`${name}.filter.sum`, () => {
            _iterator.filter(transformFns[0]).sum();
          });
          bench(`${name}.filter 2`, () => {
            testIteration(_iterator.filter(transformFns[1]));
          });
          bench(`${name}.filter.reduce 2`, () => {
            _iterator.filter(transformFns[1]).reduce(reducer, 0);
          });
          bench(`${name}.filter.sum 2`, () => {
            _iterator.filter(transformFns[1]).sum();
          });
          bench(`${name}.filter.filter`, () => {
            testIteration(
              _iterator.filter(transformFns[0]).filter(transformFns[1])
            );
          });
          bench(`${name}.filter.filter.reduce`, () => {
            _iterator
              .filter(transformFns[0])
              .filter(transformFns[1])
              .reduce(reducer, 0);
          });
          bench(`${name}.filter.filter.sum`, () => {
            _iterator.filter(transformFns[0]).filter(transformFns[1]).sum();
          });
        }
        bench(`Array.filter`, () => {
          testIteration(array.filter(transformFns[0]));
        });
        bench(`Array.filter.reduce`, () => {
          array.filter(transformFns[0]).reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.filter`, () => {
          testIteration(metarhiaIterator().filter(transformFns[0]));
        });
        bench(`MetarhiaIterator.filter.reduce`, () => {
          metarhiaIterator().filter(transformFns[0]).reduce(reducer, 0);
        });
        bench(`Array.filter 2`, () => {
          testIteration(array.filter(transformFns[1]));
        });
        bench(`Array.filter.reduce 2`, () => {
          array.filter(transformFns[1]).reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.filter 2`, () => {
          testIteration(metarhiaIterator().filter(transformFns[1]));
        });
        bench(`MetarhiaIterator.filter.reduce 2`, () => {
          metarhiaIterator().filter(transformFns[1]).reduce(reducer, 0);
        });
        bench(`Array.filter.filter`, () => {
          testIteration(array.filter(transformFns[0]).filter(transformFns[1]));
        });
        bench(`Array.filter.filter.reduce`, () => {
          array
            .filter(transformFns[0])
            .filter(transformFns[1])
            .reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.filter.filter`, () => {
          testIteration(
            metarhiaIterator().filter(transformFns[0]).filter(transformFns[1])
          );
        });
        bench(`MetarhiaIterator.filter.filter.reduce`, () => {
          metarhiaIterator()
            .filter(transformFns[0])
            .filter(transformFns[1])
            .reduce(reducer, 0);
        });
      });
    } else if (transformName === 'map') {
      describe.concurrent(`iteration map - ${size}`, () => {
        for (let [name, _iterator] of iterators) {
          bench(`${name}.map`, () => {
            testIteration(_iterator.map(transformFns[0]));
          });
          bench(`${name}.map.reduce`, () => {
            _iterator.map(transformFns[0]).reduce(reducer, 0);
          });
          bench(`${name}.map.sum`, () => {
            _iterator.map(transformFns[0]).sum();
          });
          bench(`${name}.map 2`, () => {
            testIteration(_iterator.map(transformFns[1]));
          });
          bench(`${name}.map.reduce 2`, () => {
            _iterator.map(transformFns[1]).reduce(reducer, 0);
          });
          bench(`${name}.map.sum 2`, () => {
            _iterator.map(transformFns[1]).sum();
          });
          bench(`${name}.map.map`, () => {
            testIteration(_iterator.map(transformFns[0]).map(transformFns[1]));
          });
          bench(`${name}.map.map.reduce`, () => {
            _iterator
              .map(transformFns[0])
              .map(transformFns[1])
              .reduce(reducer, 0);
          });
          bench(`${name}.map.map.sum`, () => {
            _iterator.map(transformFns[0]).map(transformFns[1]).sum();
          });
        }
        bench(`Array.map`, () => {
          testIteration(array.map(transformFns[0]));
        });
        bench(`Array.map.reduce`, () => {
          array.map(transformFns[0]).reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.map`, () => {
          testIteration(metarhiaIterator().map(transformFns[0]));
        });
        bench(`MetarhiaIterator.map.reduce`, () => {
          metarhiaIterator().map(transformFns[0]).reduce(reducer, 0);
        });
        bench(`Array.map 2`, () => {
          testIteration(array.map(transformFns[1]));
        });
        bench(`Array.map.reduce 2`, () => {
          array.map(transformFns[1]).reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.map 2`, () => {
          testIteration(metarhiaIterator().map(transformFns[1]));
        });
        bench(`MetarhiaIterator.map.reduce 2`, () => {
          metarhiaIterator().map(transformFns[1]).reduce(reducer, 0);
        });
        bench(`Array.map.map`, () => {
          testIteration(array.map(transformFns[0]).map(transformFns[1]));
        });
        bench(`Array.map.map.reduce`, () => {
          array.map(transformFns[0]).map(transformFns[1]).reduce(reducer, 0);
        });
        bench(`MetarhiaIterator.map.map`, () => {
          testIteration(
            metarhiaIterator().map(transformFns[0]).map(transformFns[1])
          );
        });
        bench(`MetarhiaIterator.map.map.reduce`, () => {
          metarhiaIterator()
            .map(transformFns[0])
            .map(transformFns[1])
            .reduce(reducer, 0);
        });
      });
    }
  }
}
